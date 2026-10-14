// Qwen3.6's row glue (glue.py, qmm.py group sums, attention.py paths) with Triton's bits: same element-to-thread order.

#include <cuda_bf16.h>
#include <stdint.h>

#include "pdl.cuh"

namespace {

constexpr float LOG2E = 1.4426950408889634f;  // Triton's tl.exp scales by this fp32 constant, then ex2.approx
constexpr int D = 2048;                       // add_rmsnorm's row: 256 threads of 8 values
constexpr int DV = 128;                       // gated_norm's value head: 32 threads of 4 values
constexpr int GD = 256;                       // gate_mul's head: 64 threads of 4 values
constexpr int MAXD = 128;                     // paths' deepest root-to-row chain

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Triton's tl.sqrt: sqrt.approx.f32.
__device__ __forceinline__ float sqrt_approx(float x) {
    float y;
    asm("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Triton's fp32 '/': div.full.f32, not the correctly rounded quotient.
__device__ __forceinline__ float divf(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint32_t bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint32_t pack(float a, float b) { return bits(a) | (bits(b) << 16); }

// tl.sigmoid: 1 / (1 + exp(-x)).
__device__ __forceinline__ float sigmoid_tl(float x) { return divf(1.0f, __fadd_rn(1.0f, ex2(__fmul_rn(-x, LOG2E)))); }

// 1 / sqrt(sum / n + eps) with the sum's multiply by 1 / n fused into the add of eps, as Triton's FFMA.
template <int N>
__device__ __forceinline__ float inv_rms(float sum, float eps) { return divf(1.0f, sqrt_approx(__fmaf_rn(sum, 1.0f / N, eps))); }

// A warp's lanes summed by xor butterfly from `first` down to 1 (every lane ends with the same sum).
template <int FIRST>
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int off = FIRST; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// sum of squares of a thread's values in Triton's order: (v0^2 + v1^2) first, then v2^2 ...
template <int N>
__device__ __forceinline__ float squares(const float (&v)[N]) {
    float s = __fmaf_rn(v[0], v[0], __fmul_rn(v[1], v[1]));
#pragma unroll
    for (int i = 2; i < N; ++i) s = __fmaf_rn(v[i], v[i], s);
    return s;
}

// a thread's values summed left to right
template <int N>
__device__ __forceinline__ float chain(const float (&v)[N]) {
    float s = __fadd_rn(v[0], v[1]);
#pragma unroll
    for (int i = 2; i < N; ++i) s = __fadd_rn(s, v[i]);
    return s;
}

// h = bf16(x + r) (or x), y = bf16(h * rsqrt * w), xs = y's 64-group sums; a block a row of 8 values per thread.
template <bool HAS_R>
__device__ __forceinline__ void add_rms(const uint16_t* X, const uint16_t* R, const uint16_t* W, uint16_t* H, uint16_t* Y, float* XS, float eps) {
    __shared__ float parts[8];
    tf_grid_wait();
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D + 8 * t;
    const uint4 xv = *reinterpret_cast<const uint4*>(X + at);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
    float x[8];
    if constexpr (HAS_R) {
        const uint4 rv = *reinterpret_cast<const uint4*>(R + at);
        const uint32_t rw[4] = {rv.x, rv.y, rv.z, rv.w};
        uint4 hv;
        uint32_t* hw = reinterpret_cast<uint32_t*>(&hv);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float a = rbf(__fadd_rn(lo(xw[i]), lo(rw[i]))), b = rbf(__fadd_rn(hi(xw[i]), hi(rw[i])));
            x[2 * i] = a;
            x[2 * i + 1] = b;
            hw[i] = pack(a, b);
        }
        *reinterpret_cast<uint4*>(H + at) = hv;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            x[2 * i] = lo(xw[i]);
            x[2 * i + 1] = hi(xw[i]);
        }
    }
    float ss = butterfly<16>(squares(x));
    if (lane == 0) parts[warp] = ss;
    __syncthreads();
    ss = lane < 8 ? parts[lane] : 0.0f;  // the eight warp sums by xor 4, 2, 1
    ss = __shfl_sync(0xffffffffu, butterfly<4>(ss), 0);
    const float inv = inv_rms<D>(ss, eps);
    const uint4 wv = *reinterpret_cast<const uint4*>(W + 8 * t);
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
    float y[8];
    uint4 yv;
    uint32_t* yw = reinterpret_cast<uint32_t*>(&yv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[2 * i] = rbf(__fmul_rn(__fmul_rn(x[2 * i], inv), lo(ww[i])));
        y[2 * i + 1] = rbf(__fmul_rn(__fmul_rn(x[2 * i + 1], inv), hi(ww[i])));
        yw[i] = pack(y[2 * i], y[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(Y + at) = yv;
    const float g = butterfly<4>(chain(y));
    if ((t & 7) == 0) XS[static_cast<int64_t>(blockIdx.x) * (D / 64) + (t >> 3)] = g;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) tf_qwen_add_rmsnorm(const uint16_t* X, const uint16_t* W, uint16_t* Y, float* XS, float eps) {
    add_rms<false>(X, X, W, nullptr, Y, XS, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_qwen_add_rmsnorm_res(const uint16_t* X, const uint16_t* R, const uint16_t* W, uint16_t* H, uint16_t* Y, float* XS, float eps) {
    add_rms<true>(X, R, W, H, Y, XS, eps);
}

// Block (row, value head), a warp of 4 values per thread: out = bf16(z * sigmoid(z) * (y * rsqrt * w)), xs = sums.
extern "C" __global__ void __launch_bounds__(32) tf_qwen_gated_norm(const uint16_t* Y, const uint16_t* Z, const uint16_t* W, uint16_t* OUT, float* XS, float eps, int vh) {
    tf_grid_wait();
    const int t = threadIdx.x;
    const int64_t head = static_cast<int64_t>(blockIdx.x) * vh + blockIdx.y;
    const int64_t at = head * DV + 4 * t;
    const uint2 yv = *reinterpret_cast<const uint2*>(Y + at), zv = *reinterpret_cast<const uint2*>(Z + at), wv = *reinterpret_cast<const uint2*>(W + 4 * t);
    const float y[4] = {lo(yv.x), hi(yv.x), lo(yv.y), hi(yv.y)};
    const float z[4] = {lo(zv.x), hi(zv.x), lo(zv.y), hi(zv.y)};
    const float w[4] = {lo(wv.x), hi(wv.x), lo(wv.y), hi(wv.y)};
    const float inv = inv_rms<DV>(butterfly<16>(squares(y)), eps);
    float o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = rbf(__fmul_rn(__fmul_rn(z[i], sigmoid_tl(z[i])), __fmul_rn(__fmul_rn(y[i], inv), w[i])));
    *reinterpret_cast<uint2*>(OUT + at) = make_uint2(pack(o[0], o[1]), pack(o[2], o[3]));
    const float g = butterfly<8>(chain(o));
    if ((t & 15) == 0) XS[head * (DV / 64) + (t >> 4)] = g;
}

// Block (row, head), 64 threads of 4 values: out = bf16(o * sigmoid(gate)) from the [q | gate] rows, xs = group sums.
extern "C" __global__ void __launch_bounds__(64) tf_qwen_gate_mul(const uint16_t* O, const uint16_t* QG, uint16_t* OUT, float* XS, int heads) {
    tf_grid_wait();
    const int t = threadIdx.x;
    const int64_t head = static_cast<int64_t>(blockIdx.x) * heads + blockIdx.y;
    const int64_t at = head * GD + 4 * t;
    const uint2 ov = *reinterpret_cast<const uint2*>(O + at), gv = *reinterpret_cast<const uint2*>(QG + head * 2 * GD + GD + 4 * t);
    const float o[4] = {lo(ov.x), hi(ov.x), lo(ov.y), hi(ov.y)};
    const float g[4] = {lo(gv.x), hi(gv.x), lo(gv.y), hi(gv.y)};
    float r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = rbf(__fmul_rn(o[i], sigmoid_tl(g[i])));
    *reinterpret_cast<uint2*>(OUT + at) = make_uint2(pack(r[0], r[1]), pack(r[2], r[3]));
    const float s = butterfly<8>(chain(r));
    if ((t & 15) == 0) XS[head * (GD / 64) + (t >> 4)] = s;
}

// Block (row, 16 groups), 8 threads of 8 values a group: xs[row][group] = the group's fp32 sum, rows strided by ldx.
extern "C" __global__ void __launch_bounds__(128) tf_qwen_group_sums(const uint16_t* X, float* XS, int ldx, int kg) {
    tf_grid_wait();
    const int t = threadIdx.x, group = blockIdx.y * 16 + (t >> 3), j = t & 7;
    float s = 0.0f;
    if (group < kg) {
        const uint4 v = *reinterpret_cast<const uint4*>(X + static_cast<int64_t>(blockIdx.x) * ldx + group * 64 + 8 * j);
        const float e[8] = {lo(v.x), hi(v.x), lo(v.y), hi(v.y), lo(v.z), hi(v.z), lo(v.w), hi(v.w)};
        s = chain(e);
    }
    s = butterfly<4>(s);
    if (j == 0 && group < kg) XS[static_cast<int64_t>(blockIdx.x) * kg + group] = s;
}

// A thread a node: its root-to-node chain of window rows (parents are rows, -1 at a root), up to 128 deep, and depth.
extern "C" __global__ void __launch_bounds__(128) tf_qwen_paths(const int* PARENTS, int* PATHS, int* DEPTHS, int rows) {
    tf_grid_wait();
    const int node = blockIdx.x * 128 + threadIdx.x;
    if (node >= rows) return;
    int cur = node, depth = 0;
    while (cur >= 0 && depth < MAXD) {
        ++depth;
        cur = PARENTS[cur];
    }
    DEPTHS[node] = depth;
    cur = node;
    for (int slot = depth - 1; slot >= 0; --slot) {
        PATHS[static_cast<int64_t>(node) * MAXD + slot] = cur;
        if (slot > 0) cur = PARENTS[cur];
    }
}
