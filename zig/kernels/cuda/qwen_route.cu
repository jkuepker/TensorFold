// The Qwen3.6 MoE router: fp32 logits on tensor cores, softmax top-k with a sigmoid-gated shared slot, the slot-order combine.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

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

// Warp (16-row tile, 8 experts of a 32-expert block): OUT[m][e] = x[m] . w[e], mma.m16n8k16 chained over K in order.
extern "C" __global__ void __launch_bounds__(128) tf_qwen_router(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                   float* __restrict__ OUT, int M, int x_stride, int D, int NE) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 16, e0 = (blockIdx.y * 4 + warp) * 8;
    const bool ra = row0 + g < M, rb = row0 + g + 8 < M, eb = e0 + g < NE;
    const uint32_t* x0 = reinterpret_cast<const uint32_t*>(X + static_cast<int64_t>(ra ? row0 + g : 0) * x_stride) + t;
    const uint32_t* x1 = reinterpret_cast<const uint32_t*>(X + static_cast<int64_t>(rb ? row0 + g + 8 : 0) * x_stride) + t;
    const uint32_t* w0 = reinterpret_cast<const uint32_t*>(W + static_cast<int64_t>(eb ? e0 + g : 0) * D) + t;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    if (e0 < NE) {
#pragma unroll 8
        for (int k = 0; k < D / 2; k += 8) {  // 16 columns a step, in 32-bit words
            const uint32_t a0 = ra ? x0[k] : 0u, a1 = rb ? x1[k] : 0u, a2 = ra ? x0[k + 4] : 0u, a3 = rb ? x1[k + 4] : 0u;
            const uint32_t b0 = eb ? w0[k] : 0u, b1 = eb ? w0[k + 4] : 0u;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                         : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    const int e = e0 + 2 * t;
    if (ra && e < NE) OUT[static_cast<int64_t>(row0 + g) * NE + e] = c0;
    if (ra && e + 1 < NE) OUT[static_cast<int64_t>(row0 + g) * NE + e + 1] = c1;
    if (rb && e < NE) OUT[static_cast<int64_t>(row0 + g + 8) * NE + e] = c2;
    if (rb && e + 1 < NE) OUT[static_cast<int64_t>(row0 + g + 8) * NE + e + 1] = c3;
}

// One warp a row: top-k by logit (lower id on ties), weights exp(l_k - l_0) / sum as bf16, then the shared slot's sigmoid gate.
extern "C" __global__ void __launch_bounds__(32) tf_qwen_topk(const float* __restrict__ L, int* __restrict__ PICK, float* __restrict__ WTS, int NE,
                                                                int NL, int TOPK) {
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

template <typename T> __device__ __forceinline__ float load_y(const T* p);
template <> __device__ __forceinline__ float load_y<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float load_y<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// Block (row, 512 columns): the slots summed in slot order as fp32 w * y fused multiply-adds, rounded once to bf16.
template <typename T>
__device__ __forceinline__ void combine(const T* __restrict__ Y, const float* __restrict__ W, __nv_bfloat16* __restrict__ OUT, int S, int D) {
    const int64_t r = blockIdx.x;
    const int d = blockIdx.y * 512 + threadIdx.x * 2;
    for (int j = 0; j < 2; ++j) {
        if (d + j >= D) return;
        float acc = 0.0f;
        for (int k = 0; k < S; ++k) acc = __fmaf_rn(W[r * S + k], load_y<T>(Y + (r * S + k) * D + d + j), acc);
        OUT[r * D + d + j] = __float2bfloat16_rn(acc);
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) tf_qwen_combine_f32(const float* Y, const float* W, __nv_bfloat16* OUT, int S, int D) {
    combine<float>(Y, W, OUT, S, D);
}

extern "C" __global__ void __launch_bounds__(256) tf_qwen_combine_bf16(const __nv_bfloat16* Y, const float* W, __nv_bfloat16* OUT, int S, int D) {
    combine<__nv_bfloat16>(Y, W, OUT, S, D);
}
