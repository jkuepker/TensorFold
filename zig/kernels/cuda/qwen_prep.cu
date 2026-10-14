// Qwen3.6's prep kernels (glue.py _gdn_pre, _attn_prep) op for op as Triton ran them, fused fmas written out.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "pdl.cuh"

namespace {

constexpr float LOG2E = 1.4426950408889634f;  // tl.exp scales by this fp32 constant, then ex2.approx

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

// tl.sqrt: sqrt.approx.f32.
__device__ __forceinline__ float sqrtf_tl(float x) {
    float y;
    asm("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float expf_tl(float x) { return ex2(__fmul_rn(x, LOG2E)); }

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint16_t bf_bits(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }

// 1 / (1 + exp(-x)), tl.sigmoid.
__device__ __forceinline__ float sigmoid_tl(float x) { return divf(1.0f, __fadd_rn(1.0f, expf_tl(__fsub_rn(0.0f, x)))); }

// libdevice's logf as Triton's tl.log runs it: x = 2^e (1 + m) split at 2/3, a degree-9 polynomial in m.
__device__ __forceinline__ float logf_tl(float x) {
    const int ix = __float_as_int(x);
    const int e = (ix - 0x3f2aaaab) & static_cast<int>(0xff800000u);
    const float m = __fsub_rn(__int_as_float(ix - e), 1.0f);
    float p = __fmaf_rn(m, -__int_as_float(0x3e055027), 0.14084610342979431152f);
    p = __fmaf_rn(m, p, -0.12148627638816833496f);
    p = __fmaf_rn(m, p, 0.13980610668659210205f);
    p = __fmaf_rn(m, p, -0.16684235632419586182f);
    p = __fmaf_rn(m, p, 0.20012299716472625732f);
    p = __fmaf_rn(m, p, -0.24999669194221496582f);
    p = __fmaf_rn(m, p, 0.33333182334899902344f);
    p = __fmaf_rn(m, p, -0.5f);
    p = __fmul_rn(m, p);
    p = __fmaf_rn(m, p, m);
    float r = __fmaf_rn(__fmul_rn(static_cast<float>(e), 1.1920928955078125e-07f), 0.69314718246459960938f, p);
    if (static_cast<uint32_t>(ix) >= 0x7f800000u) r = __fmaf_rn(x, __int_as_float(0x7f800000), __int_as_float(0x7f800000));
    return x != 0.0f ? r : -__int_as_float(0x7f800000);
}

// A 64-thread block's sum: each warp by xor butterfly (16, 8, 4, 2, 1), then warp 0 plus warp 1.
__device__ __forceinline__ float block64_sum(float v, float* part) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    return __fadd_rn(part[0], part[1]);
}

}  // namespace

// Program (row, head), 64 threads of 2 channels: 4-tap conv over [state; qkv], SiLU, q/k norm, v, g and beta.
extern "C" __global__ void __launch_bounds__(64)
    tf_qwen_gdn_pre(const uint16_t* __restrict__ QKV, const uint16_t* __restrict__ CS, const uint16_t* __restrict__ CW,
                    const int* __restrict__ WIN, const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
                    const float* __restrict__ ALOG, const float* __restrict__ DTB, uint16_t* __restrict__ Q,
                    uint16_t* __restrict__ K, uint16_t* __restrict__ V, float* __restrict__ G, float* __restrict__ BETA,
                    const int* __restrict__ SID, int C, int KH, int VH, int NKEEP, int MULTI) {
    constexpr int DK = 128;
    __shared__ float part[2];
    tf_grid_wait();
    tf_grid_launch();
    const long long row = blockIdx.x;
    const int head = blockIdx.y;
    const long long ch = static_cast<long long>(head) * DK + 2 * threadIdx.x;
    const long long cs_row = MULTI ? static_cast<long long>(SID[row]) * NKEEP : 0;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int j = 0; j <= NKEEP; ++j) {
        const long long src = WIN[row * (NKEEP + 1) + j];
        const uint16_t* x = src < NKEEP ? CS + (cs_row + src) * C + ch : QKV + (src - NKEEP) * C + ch;
        const uint16_t* w = CW + ch * (NKEEP + 1) + j;
        acc0 = __fmaf_rn(bf(x[0]), bf(w[0]), acc0);
        acc1 = __fmaf_rn(bf(x[1]), bf(w[NKEEP + 1]), acc1);
    }
    const float c0 = bf16_round(__fmul_rn(acc0, sigmoid_tl(acc0)));
    const float c1 = bf16_round(__fmul_rn(acc1, sigmoid_tl(acc1)));
    if (head < 2 * KH) {
        const float s = block64_sum(__fmaf_rn(c0, c0, __fmul_rn(c1, c1)), part);
        const float inv = divf(1.0f, sqrtf_tl(__fmaf_rn(s, 1.0f / DK, 1e-6f)));
        const float scale = head < KH ? 1.0f / DK : 0.08838834764831845f;  // 1 / sqrt(128), rounded to fp32
        const long long base = (row * KH + (head < KH ? head : head - KH)) * DK + 2 * threadIdx.x;
        uint16_t* out = head < KH ? Q + base : K + base;
        out[0] = bf_bits(__fmul_rn(__fmul_rn(c0, inv), scale));
        out[1] = bf_bits(__fmul_rn(__fmul_rn(c1, inv), scale));
        return;
    }
    const int hv = head - 2 * KH;
    uint16_t* v = V + (row * VH + hv) * DK + 2 * threadIdx.x;
    v[0] = bf_bits(c0);
    v[1] = bf_bits(c1);
    if (threadIdx.x != 0) return;
    const float a = __fadd_rn(bf(A[row * VH + hv]), DTB[hv]);
    const float sp = a > 20.0f ? a : logf_tl(__fadd_rn(1.0f, expf_tl(a)));
    const float g = expf_tl(__fmul_rn(sp, __fsub_rn(0.0f, expf_tl(ALOG[hv]))));
    G[row * VH + hv] = g;
    BETA[row * VH + hv] = sigmoid_tl(bf(B[row * VH + hv]));
}

// Program (row, head), 64 threads of 4 elements (D = 256): q heads from [q | gate] rows, then k heads; norm, rotary.
extern "C" __global__ void __launch_bounds__(64)
    tf_qwen_attn_prep(const uint16_t* __restrict__ QG, const uint16_t* __restrict__ KV, const uint16_t* __restrict__ QN,
                      const uint16_t* __restrict__ KN, const int* __restrict__ POS, const float* __restrict__ INV,
                      uint16_t* __restrict__ QOUT, uint16_t* __restrict__ KOUT, float eps, int H, int HKV, int D, int HALF) {
    __shared__ float part[2];
    tf_grid_wait();
    tf_grid_launch();
    const long long row = blockIdx.x;
    const int head = blockIdx.y;
    const bool is_q = head < H;
    const uint16_t* src = is_q ? QG + (row * H + head) * 2 * D : KV + (row * HKV + head - H) * D;
    const uint16_t* wt = is_q ? QN : KN;
    const int d0 = 4 * threadIdx.x;
    const int p0 = d0 < HALF ? d0 + HALF : (d0 < 2 * HALF ? d0 - HALF : d0);  // rotate-half partner of d0 (blocks of 4 stay whole)
    float x[4], xp[4], w[4], wp[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] = bf(src[d0 + i]);
        w[i] = bf(wt[d0 + i]);
        xp[i] = bf(src[p0 + i]);
        wp[i] = bf(wt[p0 + i]);
    }
    const float s = block64_sum(__fmaf_rn(x[3], x[3], __fmaf_rn(x[2], x[2], __fmaf_rn(x[0], x[0], __fmul_rn(x[1], x[1])))), part);
    const float inv = divf(1.0f, sqrtf_tl(__fmaf_rn(s, 1.0f / 256.0f, eps)));
    float out[4];
    if (d0 < 2 * HALF) {
        const float pos = static_cast<float>(POS[row]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float xn = bf16_round(__fmul_rn(__fmul_rn(x[i], inv), w[i]));
            const float xpn = bf16_round(__fmul_rn(__fmul_rn(xp[i], inv), wp[i]));
            const float ang = __fmul_rn(pos, INV[d0 < HALF ? d0 + i : d0 + i - HALF]);
            const float cs = cosf(ang), sn = sinf(ang);
            const float t = __fmul_rn(xpn, sn);
            out[i] = d0 < HALF ? __fmaf_rn(xn, cs, -t) : __fmaf_rn(xn, cs, t);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = bf16_round(__fmul_rn(__fmul_rn(x[i], inv), w[i]));
    }
    uint16_t* dst = is_q ? QOUT + (row * H + head) * D : KOUT + (row * HKV + head - H) * D;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[d0 + i] = bf_bits(out[i]);
}
