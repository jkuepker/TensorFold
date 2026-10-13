// Qwen3.6 MoE tree attention (attention.py's _shared, _tail and _merge): Triton's bits from native CUDA.
// Both dots are mma.m16n8k16 chains in key and head-dim order, softmax is ex2.approx of x * log2(e), l is fma(l, alpha, row sum).

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "pdl.cuh"

namespace {

typedef __nv_bfloat16 bf16;

constexpr int H = 16, HK = 2, D = 256, G = H / HK;  // query heads, key heads, head dim, query heads a key head
constexpr int CH = 512, GR = 4, MAXD = 128;         // keys a chunk, chunks a folded group, window depth
constexpr int KT = 64;                              // keys a tile
constexpr int LDK = D + 8, LDP = KT + 8;            // shared row strides (elements): padded against bank conflicts
constexpr float LOG2E = 1.4426950408889634f;        // Triton's tl.exp scales by this fp32 constant, then ex2.approx
constexpr float SCALE = 0.0625f;                    // head_dim ** -0.5

struct Smem {
    bf16 k[KT * LDK];
    bf16 v[KT * LDK];
    bf16 p[16 * LDP];
    float redm[4][16];  // each warp's row maxima
    float reds[4][16];  // each warp's row sums
};

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float expf_tl(float x) { return ex2(__fmul_rn(x, LOG2E)); }

__device__ __forceinline__ float maxf(float a, float b) {
    float y;
    asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

// Triton's fp32 '/': div.full.f32, not the correctly rounded quotient.
__device__ __forceinline__ float divf(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    uint32_t y;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(hi), "f"(lo));
    return y;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int n = valid ? 16 : 0;  // zero fill: Triton's masked loads read 0
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(n));
}

// A tile's 64 key (or value) rows, one warp a row at a time; `row(r)` is the row's source, null for a zero row.
template <class F> __device__ __forceinline__ void issue(bf16* dst, const bf16* any, int w, int lane, F row) {
#pragma unroll 4
    for (int j = 0; j < 16; ++j) {
        const int r = w + 4 * j;
        const bf16* s = row(r);
        cp16(dst + r * LDK + lane * 8, s ? s + lane * 8 : any, s != nullptr);
    }
    asm volatile("cp.async.commit_group;");
}

// One program's running softmax state for its 16 query rows: this thread's rows g and g + 8, and its 8 n-tiles of o.
struct Acc {
    float m[2], l[2], o[8][4];
};

__device__ __forceinline__ void clear(Acc& a) {
    a.m[0] = a.m[1] = -INFINITY;
    a.l[0] = a.l[1] = 0.0f;
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int c = 0; c < 4; ++c) a.o[n][c] = 0.0f;
}

// _fold's weights: the old state times a, the new partial times b, under the new max.
struct Coef {
    float a, b, m;
};

__device__ __forceinline__ Coef coef(float m, float cm, float cl) {
    const bool active = cl > 0.0f;
    Coef c;
    c.m = active ? maxf(m, cm) : m;
    c.a = active ? (m == -INFINITY ? 0.0f : expf_tl(__fsub_rn(m, c.m))) : 1.0f;
    c.b = active ? expf_tl(__fsub_rn(cm, c.m)) : 0.0f;
    return c;
}

// attention._fold into `g`: l * a + cl * b and o * a + co * b are one fma over the rounded product.
__device__ __forceinline__ void fold(Acc& g, const Acc& c) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const Coef k = coef(g.m[q], c.m[q], c.l[q]);
        g.l[q] = __fmaf_rn(g.l[q], k.a, __fmul_rn(c.l[q], k.b));
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            g.o[n][2 * q] = __fmaf_rn(g.o[n][2 * q], k.a, __fmul_rn(c.o[n][2 * q], k.b));
            g.o[n][2 * q + 1] = __fmaf_rn(g.o[n][2 * q + 1], k.a, __fmul_rn(c.o[n][2 * q + 1], k.b));
        }
        g.m[q] = k.m;
    }
}

// The q rows of an mma A operand: this thread's pairs of row g and g + 8 (null row: zeros, a masked load).
__device__ __forceinline__ void load_q(uint32_t (&qf)[16][4], const bf16* lo, const bf16* hi, int t4) {
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
        qf[ks][0] = lo ? *reinterpret_cast<const uint32_t*>(lo + ks * 16 + 2 * t4) : 0u;
        qf[ks][1] = hi ? *reinterpret_cast<const uint32_t*>(hi + ks * 16 + 2 * t4) : 0u;
        qf[ks][2] = lo ? *reinterpret_cast<const uint32_t*>(lo + ks * 16 + 8 + 2 * t4) : 0u;
        qf[ks][3] = hi ? *reinterpret_cast<const uint32_t*>(hi + ks * 16 + 8 + 2 * t4) : 0u;
    }
}

// attention._tile over `tiles` 64-key tiles. `src` gives the tile rows (krow, vrow: null is a zero row) and how many leading keys of a
// tile count (valid). Scores are the mma chain over the head dim, p the bf16 tile, o an mma chain onto o * alpha; the row max and sum
// meet across the four warps as the Triton program's reduction does (the warp w holds keys 8w..8w+7 and 32+8w..32+8w+7).
template <class Src> __device__ void attend(Smem& sm, Acc& a, const uint32_t (&qf)[16][4], int tiles, const Src& src) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
    const int mj = lane >> 3, mi = lane & 7;  // ldmatrix: this lane's matrix and row
    if (tiles > 0) issue(sm.k, src.any, w, lane, [&](int r) { return src.krow(0, r); });
    for (int t = 0; t < tiles; ++t) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        issue(sm.v, src.any, w, lane, [&](int r) { return src.vrow(t, r); });
        float s[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) s[j][c] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            uint32_t b[4];
            ldsm4(b, sm.k + (32 * (mj >> 1) + 8 * w + mi) * LDK + ks * 16 + 8 * (mj & 1));
            mma(s[0], qf[ks], b[0], b[1]);
            mma(s[1], qf[ks], b[2], b[3]);
        }
        const int count = src.valid(t);
        float tm[2] = {-INFINITY, -INFINITY};
        bool ok[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                ok[j][c] = 32 * j + 8 * w + 2 * t4 + (c & 1) < count;
                s[j][c] = ok[j][c] ? __fmul_rn(s[j][c], SCALE) : -INFINITY;
                tm[c >> 1] = maxf(tm[c >> 1], s[j][c]);
            }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            tm[q] = maxf(tm[q], __shfl_xor_sync(0xffffffffu, tm[q], 2));
            tm[q] = maxf(tm[q], __shfl_xor_sync(0xffffffffu, tm[q], 1));
            if (t4 == 0) sm.redm[w][g + 8 * q] = tm[q];
        }
        __syncthreads();
        if (t + 1 < tiles) issue(sm.k, src.any, w, lane, [&](int r) { return src.krow(t + 1, r); });
        else asm volatile("cp.async.commit_group;");
        float alpha[2], nm[2];
        bool active[2];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            float m = maxf(maxf(sm.redm[0][g + 8 * q], sm.redm[1][g + 8 * q]), maxf(sm.redm[2][g + 8 * q], sm.redm[3][g + 8 * q]));
            active[q] = m != -INFINITY;
            nm[q] = active[q] ? maxf(a.m[q], m) : a.m[q];
            alpha[q] = active[q] ? (a.m[q] == -INFINITY ? 0.0f : expf_tl(__fsub_rn(a.m[q], nm[q]))) : 1.0f;
        }
        float p[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) p[j][c] = ok[j][c] && active[c >> 1] ? expf_tl(__fsub_rn(s[j][c], nm[c >> 1])) : 0.0f;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int j = 0; j < 2; ++j) *reinterpret_cast<uint32_t*>(sm.p + (g + 8 * q) * LDP + 32 * j + 8 * w + 2 * t4) = pack_bf16(p[j][2 * q], p[j][2 * q + 1]);
            float v = __fadd_rn(__fadd_rn(__fadd_rn(p[0][2 * q], p[0][2 * q + 1]), p[1][2 * q]), p[1][2 * q + 1]);
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
            v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
            if (t4 == 0) sm.reds[w][g + 8 * q] = v;
        }
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const float sum = __fadd_rn(__fadd_rn(sm.reds[0][g + 8 * q], sm.reds[2][g + 8 * q]), __fadd_rn(sm.reds[1][g + 8 * q], sm.reds[3][g + 8 * q]));
            a.l[q] = __fmaf_rn(a.l[q], alpha[q], sum);
            a.m[q] = nm[q];
        }
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) a.o[n][c] = __fmul_rn(a.o[n][c], alpha[c >> 1]);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t pf[4];
            ldsm4(pf, sm.p + (mi + 8 * (mj & 1)) * LDP + ks * 16 + 8 * (mj >> 1));
#pragma unroll
            for (int pr = 0; pr < 4; ++pr) {
                uint32_t b[4];
                ldsm4t(b, sm.v + (ks * 16 + 8 * (mj & 1) + mi) * LDK + 64 * w + 16 * pr + 8 * (mj >> 1));
                mma(a.o[2 * pr], pf, b[0], b[1]);
                mma(a.o[2 * pr + 1], pf, b[2], b[3]);
            }
        }
    }
}

// Writes this thread's rows of a partial: o to PO[row][d], m to PM[row], l to PL[row]; a null `lo`/`hi` row is not stored.
__device__ __forceinline__ void store_partial(const Acc& a, float* PO, float* PM, float* PL, int64_t lo, int64_t hi, int w, int g, int t4) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int64_t row = q ? hi : lo;
        if (row < 0) continue;
#pragma unroll
        for (int n = 0; n < 8; ++n) *reinterpret_cast<float2*>(PO + row * D + 64 * w + 8 * n + 2 * t4) = make_float2(a.o[n][2 * q], a.o[n][2 * q + 1]);
        if (w == 0 && t4 == 0) {
            PM[row] = a.m[q];
            PL[row] = a.l[q];
        }
    }
}

// _shared's tile rows: whole committed chunks of one stream's caches.
struct SharedSrc {
    const bf16 *kc, *vc, *any;
    int64_t first;  // the chunk's first key
    __device__ const bf16* krow(int t, int r) const { return kc + (first + 64 * t + r) * (HK * D); }
    __device__ const bf16* vrow(int t, int r) const { return vc + (first + 64 * t + r) * (HK * D); }
    __device__ int valid(int) const { return KT; }
};

}  // namespace

// Block (key head, query tile, item): an item code is a whole 4-chunk group j >= 0, or chunk -1 - j after the whole groups.
extern "C" __global__ void __launch_bounds__(128) tf_qwen_attn_shared(const bf16* __restrict__ Q, const bf16* __restrict__ KC, const bf16* __restrict__ VC,
                                                                        const int64_t* __restrict__ OFF, const int* __restrict__ STREAM, const int* __restrict__ ITEMS,
                                                                        float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL, int W) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    tf_grid_wait();
    tf_grid_launch();
    const int item = blockIdx.x / HK, hk = blockIdx.x % HK;
    const int s = ITEMS[item * 3], first = ITEMS[item * 3 + 1], code = ITEMS[item * 3 + 2];
    const int start = STREAM[s * 4], rows = STREAM[s * 4 + 1], p = STREAM[s * 4 + 2];
    const int groups = ((p + rows + CH - 1) / CH - STREAM[s * 4 + 3]) / (GR - 1);
    const bool whole = code >= 0;
    const int chunk = whole ? code * GR : groups * GR - 1 - code;
    const int n = whole ? GR : 1;
    const int slot = whole ? code : chunk - groups * (GR - 1);
    if (!((chunk + n) * CH <= p && code < groups)) return;  // a padded plan (a graph's) skips what the keys don't fill
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
    const int64_t koff = OFF[s * 2], voff = OFF[s * 2 + 1];
    int64_t base[2];
    const bf16* qrow[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int rr = first + g + 8 * q;
        const bool ok = rr < rows * G;
        const int node = start + rr / G, head = hk * G + rr % G;
        qrow[q] = ok ? Q + (static_cast<int64_t>(node) * H + head) * D : nullptr;
        base[q] = ok ? (static_cast<int64_t>(slot) * W + node) * H + head : -1;
    }
    uint32_t qf[16][4];
    load_q(qf, qrow[0], qrow[1], t4);
    Acc gr, ch;
    clear(gr);
    for (int c = 0; c < n; ++c) {
        clear(ch);
        const SharedSrc src = {KC + koff + static_cast<int64_t>(hk) * D, VC + voff + static_cast<int64_t>(hk) * D, KC, static_cast<int64_t>(chunk + c) * CH};
        attend(sm, ch, qf, CH / KT, src);
        fold(gr, ch);
    }
    store_partial(whole ? gr : ch, PO, PM, PL, base[0], base[1], w, g, t4);
}

namespace {

// _tail's tile rows: the last committed keys of a chunk, then the row's own root path in the window's keys.
struct TailSrc {
    const bf16 *kc, *vc, *kn, *vn, *any;
    const int* path;  // this row's path
    int p, depth;
    int64_t first;  // the chunk's first key
    __device__ const bf16* row(const bf16* c, const bf16* nodes, int t, int r) const {
        const int key = static_cast<int>(first) + 64 * t + r;
        if (key < p) return c + static_cast<int64_t>(key) * (HK * D);
        if (key < p + depth) return nodes + static_cast<int64_t>(path[key - p]) * (HK * D);
        return nullptr;
    }
    __device__ const bf16* krow(int t, int r) const { return row(kc, kn, t, r); }
    __device__ const bf16* vrow(int t, int r) const { return row(vc, vn, t, r); }
    __device__ int valid(int t) const { return min(max(p + depth - static_cast<int>(first) - 64 * t, 0), KT); }
};

}  // namespace

// Block (row, key head, tail chunk): the chunk's committed keys below `p` and the row's path, one partial slot per chunk.
extern "C" __global__ void __launch_bounds__(128) tf_qwen_attn_tail(const bf16* __restrict__ Q, const bf16* __restrict__ KN, const bf16* __restrict__ VN,
                                                                      const bf16* __restrict__ KC, const bf16* __restrict__ VC, const int64_t* __restrict__ OFF,
                                                                      const int* __restrict__ STREAM, const int* __restrict__ ROWS, const int* __restrict__ PATHS,
                                                                      const int* __restrict__ DEPTHS, float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                      int W) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    tf_grid_wait();
    tf_grid_launch();
    const int node = blockIdx.x, hk = blockIdx.y;
    const int s = ROWS[node];
    const int p = STREAM[s * 4 + 2];
    const int nch = (p + STREAM[s * 4 + 1] + CH - 1) / CH;
    const int groups = (nch - STREAM[s * 4 + 3]) / (GR - 1);
    const int chunk = p / CH + blockIdx.z;
    if (chunk >= nch) return;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
    const int64_t koff = OFF[s * 2], voff = OFF[s * 2 + 1];
    const int depth = DEPTHS[node];
    const bf16* qlo = g < G ? Q + (static_cast<int64_t>(node) * H + hk * G + g) * D : nullptr;
    uint32_t qf[16][4];
    load_q(qf, qlo, nullptr, t4);  // rows 8 to 15 are the 16-wide tile's padding
    const TailSrc src = {KC + koff + static_cast<int64_t>(hk) * D, VC + voff + static_cast<int64_t>(hk) * D, KN + static_cast<int64_t>(hk) * D, VN + static_cast<int64_t>(hk) * D, KC,
                         PATHS + static_cast<int64_t>(node) * MAXD, p, depth, static_cast<int64_t>(chunk) * CH};
    Acc a;
    clear(a);
    // tiles wholly past the path hold no key: their masked scores leave m, l and o as they are
    const int tiles = min(max((p + depth - chunk * CH + KT - 1) / KT, 0), CH / KT);
    attend(sm, a, qf, tiles, src);
    const int64_t row = (static_cast<int64_t>(chunk - groups * (GR - 1)) * W + node) * H + hk * G + g;
    store_partial(a, PO, PM, PL, g < G ? row : -1, -1, w, g, t4);
}

// Block (row, key head, 64 output columns): the partial slots folded in order, each group's chunks first, then o / l as bf16.
extern "C" __global__ void __launch_bounds__(128) tf_qwen_attn_merge(const float* __restrict__ PO, const float* __restrict__ PM, const float* __restrict__ PL,
                                                                       bf16* __restrict__ OUT, const int* __restrict__ STREAM, const int* __restrict__ ROWS, int W) {
    tf_grid_wait();
    tf_grid_launch();
    const int node = blockIdx.x, hk = blockIdx.y;
    const int s = ROWS[node];
    const int nslots = STREAM[s * 4 + 3];
    const int groups = ((STREAM[s * 4 + 2] + STREAM[s * 4 + 1] + CH - 1) / CH - nslots) / (GR - 1);
    const int row = threadIdx.x >> 4, head = hk * G + row;  // 8 query rows, 16 threads of four columns each
    const int d = blockIdx.z * 64 + (threadIdx.x & 15) * 4;
    float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float gm = -INFINITY, gl = 0.0f, go[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto fold1 = [](float& m, float& l, float (&o)[4], float cm, float cl, const float4& co) {
        const Coef k = coef(m, cm, cl);
        l = __fmaf_rn(l, k.a, __fmul_rn(cl, k.b));
        o[0] = __fmaf_rn(o[0], k.a, __fmul_rn(co.x, k.b));
        o[1] = __fmaf_rn(o[1], k.a, __fmul_rn(co.y, k.b));
        o[2] = __fmaf_rn(o[2], k.a, __fmul_rn(co.z, k.b));
        o[3] = __fmaf_rn(o[3], k.a, __fmul_rn(co.w, k.b));
        m = k.m;
    };
    for (int slot = 0; slot < groups; ++slot) {
        const int64_t at = (static_cast<int64_t>(slot) * W + node) * H + head;
        fold1(m, l, o, PM[at], PL[at], *reinterpret_cast<const float4*>(PO + at * D + d));
    }
    for (int slot = groups; slot < nslots; ++slot) {
        const int64_t at = (static_cast<int64_t>(slot) * W + node) * H + head;
        fold1(gm, gl, go, PM[at], PL[at], *reinterpret_cast<const float4*>(PO + at * D + d));
        if ((slot - groups + 1) % GR == 0 || slot == nslots - 1) {  // a group's last chunk, or the last one
            fold1(m, l, o, gm, gl, make_float4(go[0], go[1], go[2], go[3]));
            gm = -INFINITY, gl = 0.0f;
            go[0] = go[1] = go[2] = go[3] = 0.0f;
        }
    }
    uint2 out;
    out.x = pack_bf16(divf(o[0], l), divf(o[1], l));
    out.y = pack_bf16(divf(o[2], l), divf(o[3], l));
    *reinterpret_cast<uint2*>(OUT + (static_cast<int64_t>(node) * H + head) * D + d) = out;
}
