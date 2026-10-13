// The Qwen3.6 MoE router: fp32 logits on tensor cores, softmax top-k with a sigmoid-gated shared slot, the slot-order combine.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "pdl.cuh"

namespace {

constexpr float LOG2E = 1.4426950408889634f;  // Triton's tl.exp scales by this fp32 constant, then ex2.approx
constexpr int MAXC = 8;                       // experts a lane holds: up to 256 experts

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Triton's fp32 '/': div.full.f32, not the correctly rounded quotient.
__device__ __forceinline__ float divf(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float expf_tl(float x) { return ex2(__fmul_rn(x, LOG2E)); }

}  // namespace

constexpr int KC = 256, STAGES = 3, PAD = 8, LD = KC + PAD;  // K columns a stage, stages in flight, row padding against bank conflicts
constexpr int ROWS_PER_TILE = 16, EXPERTS_PER_TILE = 32;
constexpr int STAGE_ELEMS = (ROWS_PER_TILE + EXPERTS_PER_TILE) * LD;

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int n = valid ? 16 : 0;  // zero fill: Triton's masked loads read 0.0
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(n));
}

// Block (16-row tile, 32 experts, 4 warps of 8): OUT[m][e] = x[m] . w[e], mma.m16n8k16 chained over K in order, K staged through shared memory.
extern "C" __global__ void __launch_bounds__(128) tf_qwen_router(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                   float* __restrict__ OUT, int M, int x_stride, int D, int NE) {
    extern __shared__ __align__(16) __nv_bfloat16 smem[];
    tf_grid_wait();
    tf_grid_launch();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * ROWS_PER_TILE, e0 = blockIdx.y * EXPERTS_PER_TILE;
    const int chunks = D / KC;
    auto issue = [&](int c) {
        __nv_bfloat16* a = smem + (c % STAGES) * STAGE_ELEMS;
        __nv_bfloat16* b = a + ROWS_PER_TILE * LD;
        for (int i = threadIdx.x; i < ROWS_PER_TILE * (KC / 8); i += 128) {
            const int r = i / (KC / 8), col = (i % (KC / 8)) * 8;
            const bool ok = row0 + r < M;
            copy16(a + r * LD + col, X + static_cast<int64_t>(ok ? row0 + r : 0) * x_stride + c * KC + col, ok);
        }
        for (int i = threadIdx.x; i < EXPERTS_PER_TILE * (KC / 8); i += 128) {
            const int r = i / (KC / 8), col = (i % (KC / 8)) * 8;
            const bool ok = e0 + r < NE;
            copy16(b + r * LD + col, W + static_cast<int64_t>(ok ? e0 + r : 0) * D + c * KC + col, ok);
        }
        asm volatile("cp.async.commit_group;");
    };
    for (int c = 0; c < STAGES - 1; ++c) {
        if (c < chunks) issue(c);
        else asm volatile("cp.async.commit_group;");
    }
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
        __syncthreads();
        if (c + STAGES - 1 < chunks) issue(c + STAGES - 1);
        else asm volatile("cp.async.commit_group;");
        const uint32_t* a = reinterpret_cast<const uint32_t*>(smem + (c % STAGES) * STAGE_ELEMS);
        const uint32_t* b = reinterpret_cast<const uint32_t*>(smem + (c % STAGES) * STAGE_ELEMS + ROWS_PER_TILE * LD) + (warp * 8 + g) * (LD / 2);
#pragma unroll
        for (int k = 0; k < KC / 2; k += 8) {  // 16 columns a step, in 32-bit words
            const uint32_t a0 = a[g * (LD / 2) + k + t], a1 = a[(g + 8) * (LD / 2) + k + t];
            const uint32_t a2 = a[g * (LD / 2) + k + t + 4], a3 = a[(g + 8) * (LD / 2) + k + t + 4];
            const uint32_t b0 = b[k + t], b1 = b[k + t + 4];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    const int e = e0 + warp * 8 + 2 * t;
    if (row0 + g < M && e < NE) OUT[static_cast<int64_t>(row0 + g) * NE + e] = c0;
    if (row0 + g < M && e + 1 < NE) OUT[static_cast<int64_t>(row0 + g) * NE + e + 1] = c1;
    if (row0 + g + 8 < M && e < NE) OUT[static_cast<int64_t>(row0 + g + 8) * NE + e] = c2;
    if (row0 + g + 8 < M && e + 1 < NE) OUT[static_cast<int64_t>(row0 + g + 8) * NE + e + 1] = c3;
}

// One warp a row: top-k by logit (lower id on ties), weights exp(l_k - l_0) / sum as bf16, then the shared slot's sigmoid gate.
extern "C" __global__ void __launch_bounds__(32) tf_qwen_topk(const float* __restrict__ L, int* __restrict__ PICK, float* __restrict__ WTS, int NE,
                                                                int NL, int TOPK) {
    tf_grid_wait();
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x;
    const int slots = TOPK + 1;
    float v[MAXC];
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int e = lane + 32 * c;
        v[c] = e < NE ? L[r * NL + e] : -INFINITY;
    }
    float top = -INFINITY;
#pragma unroll
    for (int c = 0; c < MAXC; ++c) top = fmaxf(top, v[c]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, off));
    float total = 0.0f, exs[MAXC];
    int ids[MAXC];
    for (int k = 0; k < TOPK; ++k) {
        float m = -INFINITY;
#pragma unroll
        for (int c = 0; c < MAXC; ++c) m = fmaxf(m, v[c]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        int idx = 0x7fffffff;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c < NE && v[c] == m) idx = min(idx, lane + 32 * c);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffffu, idx, off));
        if (idx == 0x7fffffff) idx = NE;  // no entry equals m (NaN logits): the first padding column
        const float ex = expf_tl(__fsub_rn(m, top));
        total = __fadd_rn(total, ex);
        if (k < MAXC) exs[k] = ex, ids[k] = idx;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c == idx) v[c] = -INFINITY;
    }
    if (lane >= slots) return;
    int id = NE;
    float w = 0.0f;
    if (lane < TOPK) {
        float ex = 0.0f;
        int pick = 0;
#pragma unroll
        for (int k = 0; k < MAXC; ++k)
            if (k == lane) ex = exs[k], pick = ids[k];
        id = pick;
        w = bf16_round(divf(ex, total));
    } else {
        const float sg = bf16_round(L[r * NL + NE]);
        w = bf16_round(divf(1.0f, __fadd_rn(expf_tl(__fsub_rn(0.0f, sg)), 1.0f)));
    }
    PICK[r * slots + lane] = id;
    WTS[r * slots + lane] = w;
}

namespace {

template <typename T> __device__ __forceinline__ void load4(const T* p, float (&v)[4]);
template <> __device__ __forceinline__ void load4<float>(const float* p, float (&v)[4]) {
    const float4 q = *reinterpret_cast<const float4*>(p);
    v[0] = q.x, v[1] = q.y, v[2] = q.z, v[3] = q.w;
}
template <> __device__ __forceinline__ void load4<__nv_bfloat16>(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 q = *reinterpret_cast<const uint2*>(p);
    v[0] = __uint_as_float(q.x << 16), v[1] = __uint_as_float(q.x & 0xFFFF0000u);
    v[2] = __uint_as_float(q.y << 16), v[3] = __uint_as_float(q.y & 0xFFFF0000u);
}

// Block (row, 512 columns), four a thread: the slots summed in slot order as fp32 w * y fused multiply-adds, rounded once to bf16.
template <typename T>
__device__ __forceinline__ void combine(const T* __restrict__ Y, const float* __restrict__ W, __nv_bfloat16* __restrict__ OUT, int S, int D) {
    tf_grid_wait();
    const int64_t r = blockIdx.x;
    const int d = blockIdx.y * 512 + threadIdx.x * 4;
    if (d >= D) return;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < S; ++k) {
        float y[4];
        load4<T>(Y + (r * S + k) * D + d, y);
        const float w = W[r * S + k];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = __fmaf_rn(w, y[j], acc[j]);
    }
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(OUT + r * D + d);
    o[0] = __floats2bfloat162_rn(acc[0], acc[1]);
    o[1] = __floats2bfloat162_rn(acc[2], acc[3]);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(128) tf_qwen_combine_f32(const float* Y, const float* W, __nv_bfloat16* OUT, int S, int D) {
    combine<float>(Y, W, OUT, S, D);
}

extern "C" __global__ void __launch_bounds__(128) tf_qwen_combine_bf16(const __nv_bfloat16* Y, const float* W, __nv_bfloat16* OUT, int S, int D) {
    combine<__nv_bfloat16>(Y, W, OUT, S, D);
}

// The MTP head's confidence: a bf16 logits row's max and sum of exp(x - max), torch.softmax's float32 normalizer (fixed tree).
extern "C" __global__ void __launch_bounds__(1024) tf_qwen_row_stats(const __nv_bfloat16* __restrict__ L, int N, float* __restrict__ OUT) {
    __shared__ float part[32];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float m = -INFINITY;
    for (int i = t; i < N; i += 1024) m = fmaxf(m, __bfloat162float(L[i]));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
    if (lane == 0) part[warp] = m;
    __syncthreads();
    m = part[lane];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
    __syncthreads();
    float s = 0.0f;
    for (int i = t; i < N; i += 1024) s += expf(__bfloat162float(L[i]) - m);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    if (lane == 0) part[warp] = s;
    __syncthreads();
    s = part[lane];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    if (t == 0) {
        OUT[0] = m;
        OUT[1] = s;
    }
}
