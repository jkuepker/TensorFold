# ROCm decode tree attention (attention_rocm.cu's shared_kernel and tail_kernel, attention.py's Triton _merge) in
# Mojo for gfx12 (wave32), launched from attention_rocm_mojo.cu with hipModuleLaunchKernel (TF_ROCM_TREE_KERNEL=mojo).
# The arithmetic is the HIP and Triton kernels', op for op, so a row gets the same bits from either:
# - a 16-key tile folds into a wave's 16 (row, head) pairs as ``fold`` does (attention_rocm.cu): S^T = K Q^T on 16
#   chained v_wmma_f32_16x16x16_bf16, scores times scale, masked keys at -inf, a maxnum running max, __expf as
#   v_exp_f32(x * log2e), p summed in key order then across the lane pair, l = fma(l, alpha, sum), the O rows
#   rescaled where alpha moved, then O += P V with P rounded to bf16 (nearest even);
# - the tiles in LDS hold the same bf16 values whether the cache holds bf16 rows or packed FP8 rows (widened exactly);
# - the merge folds a row's chunks in key order as Triton's _merge compiles: exp as llvm.exp2's range-reduced
#   lowering of x * log2e, o = fma(o, a, b * co), l = fma(cl, b, l * a), a precise division, bf16 nearest even
#   (NaN as 0x7FFF).
# Mojo contracts a * b + c into an fma by default, so every fused or unfused step here is written out explicitly.
#
# Kernels are top-level defs (one per instantiation: cuda/mojo.py names a code object after its def) and must be
# instantiated by a host function, which is what tf_attention_instantiate is for; it is never called.
from std.sys import llvm_intrinsic
from std.memory import bitcast, stack_allocation
from std.utils import StaticTuple
from max.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier, MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext

comptime F32 = DType.float32
comptime U32 = DType.uint32
comptime U16Ptr = UnsafePointer[UInt16, MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime U32Ptr = UnsafePointer[UInt32, MutAnyOrigin]
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime IPtr = UnsafePointer[Int32, MutAnyOrigin]
comptime LPtr = UnsafePointer[Int64, MutAnyOrigin]
comptime SPtr = UnsafePointer[UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]

comptime D = 256
comptime CH = 512  # keys a chunk, at fixed absolute positions (attention.CHUNK)
comptime MAXD = 128  # a path's most rows (attention.MAX_NODES)
comptime KROW = D + 8  # bf16 a K row in LDS: 8 of padding, so the 16 rows a step reads miss each other's banks
comptime VT = 16 * KROW  # bf16 offset of the transposed values in a tile
comptime TILE = (16 * KROW + D * 16) // 2  # uint32 a tile: K by rows, then V transposed (B operand rows are keys)
comptime LOADERS = 4  # loader waves a pipelined block
comptime NL = 32 * LOADERS  # loader threads
comptime ROW8 = D + 16  # a packed FP8 row: 256 e4m3 bytes, the int8 exponent, padding
comptime P16 = 16 * D // 8  # pieces of K (and of V) a bf16 tile: 8 values each
comptime P8 = 16 * D // 16  # pieces a tile over packed rows: 16 values each
comptime KV_NT = False  # nontemporal key and value loads: measured slower (attn_bench), kept off
comptime V_TR = True  # bf16 values transposed by the load (global_load_tr_b128), not by 16-bit LDS stores
comptime DEPTH = 2  # tiles a pipelined loader holds in registers (attention_rocm.cu's schedule); never a row's bits
comptime MERGE_HEADS = 4  # (row, head) pairs a merge block folds: 64 threads each, 4 columns a thread


@always_inline
def neg_inf() -> Float32:
    return bitcast[F32, 1](UInt32(0xFF800000))


@always_inline
def maxnum(a: Float32, b: Float32) -> Float32:
    return llvm_intrinsic["llvm.maxnum.f32", Float32](a, b)


# HIP's __expf: v_exp_f32 of x * log2e, no range reduction.
@always_inline
def fast_exp(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.amdgcn.exp2.f32", Float32](x * bitcast[F32, 1](UInt32(0x3FB8AA3B)))


# Triton's tl.exp on AMD: llvm.exp2(x * log2e), whose lowering scales arguments under -126 by 2^64 and back, the
# multiply fused into the scaling add (the compare reads the unfused product).
@always_inline
def triton_exp(x: Float32) -> Float32:
    var log2e = bitcast[F32, 1](UInt32(0x3FB8AA3B))
    var t = x * log2e
    var low = t < Float32(-126.0)
    var arg = x.fma(log2e, Float32(64.0) if low else Float32(0.0))
    var r = llvm_intrinsic["llvm.amdgcn.exp2.f32", Float32](arg)
    return llvm_intrinsic["llvm.ldexp.f32.i32", Float32](r, Int32(-64) if low else Int32(0))


@always_inline
def bf16_round(f: Float32) -> UInt32:
    var u = bitcast[U32, 1](f)
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


@always_inline
def pack2(a: Float32, b: Float32) -> UInt32:
    return bf16_round(a) | (bf16_round(b) << 16)


@always_inline
def wmma(a: SIMD[U32, 4], b: SIMD[U32, 4], c: SIMD[F32, 8]) -> SIMD[F32, 8]:
    return llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16", SIMD[F32, 8]](
        bitcast[DType.int16, 8](a), bitcast[DType.int16, 8](b), c
    )


@always_inline
def lane_id() -> UInt32:
    return UInt32(thread_idx.x) & 31


@always_inline
def shfl(x: Float32, src: UInt32) -> Float32:
    var r = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](Int32(src << 2), bitcast[DType.int32, 1](x))
    return bitcast[F32, 1](r)


@always_inline
def ballot(p: Bool) -> Int32:
    return llvm_intrinsic["llvm.amdgcn.ballot.i32", Int32](p)


# Loads at a 32-bit byte offset from a wave-uniform base: one global_load with the base in SGPRs (no 64-bit address
# math a piece).
@always_inline
def ld4[NT: Bool = False](p: U16Ptr, off: UInt32) -> SIMD[U32, 4]:  # 8 bf16 at element ``off``
    return (p.bitcast[UInt8]() + Int(off * 2)).bitcast[UInt32]().load[width=4, alignment=16, non_temporal=NT]()


# 8 bf16 rows of 8 lanes transposed in the load: lane j of each 8 gets element j of the 8 rows its group named
# (global_load_tr_b128), so 8 keys of one value dim arrive in one lane.
@always_inline
def ld4_tr(p: U16Ptr, off: UInt32) -> SIMD[U32, 4]:
    var at = (p.bitcast[UInt8]() + Int(off * 2)).bitcast[Int16]().address_space_cast[AddressSpace.GLOBAL]()
    return bitcast[U32, 4](llvm_intrinsic["llvm.amdgcn.global.load.tr.b128.v8i16", SIMD[DType.int16, 8]](at))


@always_inline
def ld4b[NT: Bool = False](p: U8Ptr, off: UInt32) -> SIMD[U32, 4]:  # 16 bytes at byte ``off``
    return (p + Int(off)).bitcast[UInt32]().load[width=4, alignment=16, non_temporal=NT]()


@always_inline
def uniform(x: UInt32) -> UInt32:  # lane 0's value, in an SGPR
    return UInt32(llvm_intrinsic["llvm.amdgcn.readfirstlane.i32", Int32](Int32(x)))


@always_inline
def exponent(p: U8Ptr, off: UInt32) -> Int32:  # a packed row's int8 exponent
    return Int32(bitcast[DType.int8, 1](p[Int(off)]))


# A 16-value packed piece as bf16: e4m3 to fp32, times 2^e (exact), the top 16 bits (exact: an e4m3 value has four
# significant bits); ``kv8.unpack`` in Python gives the same values.
@always_inline
def widen(raw: SIMD[U32, 4], e: Int32) -> SIMD[U32, 8]:
    var sc = llvm_intrinsic["llvm.ldexp.f32.i32", Float32](Float32(1.0), e)
    var out = SIMD[U32, 8](0)
    comptime for k in range(4):
        var w = bitcast[DType.int32, 1](raw[k])
        var a = llvm_intrinsic["llvm.amdgcn.cvt.pk.f32.fp8", SIMD[F32, 2]](w, False)
        var b = llvm_intrinsic["llvm.amdgcn.cvt.pk.f32.fp8", SIMD[F32, 2]](w, True)
        out[2 * k] = (bitcast[U32, 1](a[0] * sc) >> 16) | (bitcast[U32, 1](a[1] * sc) & 0xFFFF0000)
        out[2 * k + 1] = (bitcast[U32, 1](b[0] * sc) >> 16) | (bitcast[U32, 1](b[1] * sc) & 0xFFFF0000)
    return out


# Piece i of a bf16 tile into LDS: K by rows (key i / 32), V keys first (key i % 16) so a wave's transposed stores hit
# consecutive keys.
@always_inline
def put_k(t: SPtr, i: UInt32, x: SIMD[U32, 4]):
    (t + Int(((i >> 5) * KROW + (i & 31) * 8) >> 1)).store[alignment=16](x)


@always_inline
def put_v(t: SPtr, i: UInt32, x: SIMD[U32, 4]):
    var t16 = t.bitcast[UInt16]()
    var key = i & 15
    var col = (i >> 4) * 8
    comptime for u in range(4):
        t16[Int(VT + (col + UInt32(2 * u)) * 16 + key)] = UInt16(x[u] & 0xFFFF)
        t16[Int(VT + (col + UInt32(2 * u + 1)) * 16 + key)] = UInt16(x[u] >> 16)


# A transposed V piece (``ld4_tr``): lanes i .. i + 7 named keys i % 16 .. + 7 at dims (i / 16) 8 .. + 7, so lane i
# holds dim (i / 16) 8 + i % 8 of those 8 keys: one 16-byte store into the transposed tile.
@always_inline
def put_vt(t: SPtr, i: UInt32, x: SIMD[U32, 4]):
    (t + Int((VT + ((i >> 4) * 8 + (i & 7)) * 16 + (i & 8)) >> 1)).store[alignment=16](x)


# A 16-value piece into the tile: K by rows (key i / 16), V keys first (key i % 16).
@always_inline
def put_k16(t: SPtr, i: UInt32, x: SIMD[U32, 8]):
    var at = t + Int(((i >> 4) * KROW + (i & 15) * 16) >> 1)
    at.store[alignment=16](x.slice[4, offset=0]())
    (at + 4).store[alignment=16](x.slice[4, offset=4]())


@always_inline
def put_v16(t: SPtr, i: UInt32, x: SIMD[U32, 8]):
    var t16 = t.bitcast[UInt16]()
    var key = i & 15
    var col = (i >> 4) * 16
    comptime for u in range(8):
        t16[Int(VT + (col + UInt32(2 * u)) * 16 + key)] = UInt16(x[u] & 0xFFFF)
        t16[Int(VT + (col + UInt32(2 * u + 1)) * 16 + key)] = UInt16(x[u] >> 16)


# The cache from key ``key0`` on (``stride``: bf16 elements a key, bytes for packed rows). The 64-bit product is
# taken once a tile on wave-uniform values, so the loads' 32-bit offsets stay within a tile's rows however long the
# cache (key * stride in 32 bits wraps past 4 GiB: about 2M keys at 4 KV heads).
@always_inline
def from_key[KV8: Bool](c: U16Ptr, key0: UInt32, stride: UInt32) -> U16Ptr:
    comptime if KV8:
        return (c.bitcast[UInt8]() + Int(key0) * Int(stride)).bitcast[UInt16]()
    else:
        return c + Int(key0) * Int(stride)


# The pieces of tile ``key0`` a thread loads: bf16 (KV8 False) 8 values a piece, packed 16 bytes and the row's
# exponent; the first ``M`` of N pieces of K and as many of V, piece j at ``i0 + j * step``.
# BOUND (prompt keys): keys past ``last`` read key ``last``'s row instead of attention_rocm.cu's zeros. Their scores
# are masked to -inf and their probabilities are 0, so a finite row adds exact zeros to every sum: the same bits.
# A tile holds a key at or before ``last``, so ``last - key0`` never wraps.
@always_inline
def fetch_c[KV8: Bool, N: Int, M: Int, BOUND: Bool = False](
    mut kr: SIMD[U32, 4 * N],
    mut vr: SIMD[U32, 4 * N],
    mut ke: SIMD[DType.int32, N],
    mut ve: SIMD[DType.int32, N],
    kc: U16Ptr,
    vc: U16Ptr,
    stride: UInt32,
    key0: UInt32,
    i0: UInt32,
    step: UInt32,
    last: UInt32 = 0,
):
    var kt = from_key[KV8](kc, key0, stride)
    var vt = from_key[KV8](vc, key0, stride)
    comptime for j in range(M):
        var i = i0 + UInt32(j) * step
        var kk: UInt32  # keys from key0
        var vk: UInt32
        comptime if KV8:
            kk = i >> 4
        else:
            kk = i >> 5
        vk = i & 15
        comptime if BOUND:
            kk = min(kk, last - key0)
            vk = min(vk, last - key0)
        comptime if KV8:
            var kb = kt.bitcast[UInt8]()
            var vb = vt.bitcast[UInt8]()
            var krow = kk * stride
            var vrow = vk * stride
            kr = kr.insert[offset=4 * j](ld4b[KV_NT](kb, krow + (i & 15) * 16))
            ke[j] = exponent(kb, krow + D)
            vr = vr.insert[offset=4 * j](ld4b[KV_NT](vb, vrow + (i >> 4) * 16))
            ve[j] = exponent(vb, vrow + D)
        else:
            kr = kr.insert[offset=4 * j](ld4[KV_NT](kt, kk * stride + (i & 31) * 8))
            comptime if V_TR:
                vr = vr.insert[offset=4 * j](ld4_tr(vt, vk * stride + (i >> 4) * 8))
            else:
                vr = vr.insert[offset=4 * j](ld4[KV_NT](vt, vk * stride + (i >> 4) * 8))


@always_inline
def place_c[KV8: Bool, N: Int, M: Int](
    t: SPtr,
    kr: SIMD[U32, 4 * N],
    vr: SIMD[U32, 4 * N],
    ke: SIMD[DType.int32, N],
    ve: SIMD[DType.int32, N],
    i0: UInt32,
    step: UInt32,
):
    comptime for j in range(M):
        var i = i0 + UInt32(j) * step
        comptime if KV8:
            put_k16(t, i, widen(kr.slice[4, offset=4 * j](), ke[j]))
            put_v16(t, i, widen(vr.slice[4, offset=4 * j](), ve[j]))
        else:
            put_k(t, i, kr.slice[4, offset=4 * j]())
            comptime if V_TR:
                put_vt(t, i, vr.slice[4, offset=4 * j]())
            else:
                put_v(t, i, vr.slice[4, offset=4 * j]())


# attention_rocm.cu's ``stage`` batch: pieces i0 + j * step (j < N) below ``limit``, all loaded before any is put.
# step and limit are multiples of 32, so how many a thread takes is its wave's. bf16: one branch per count, each
# loading and placing its own pieces (registers merged across branches cost the bf16 kernel its query registers);
# packed: a branch per piece (the per-count form spills there).
@always_inline
def stage_batch[KV8: Bool, N: Int, BOUND: Bool = False](t: SPtr, kc: U16Ptr, vc: U16Ptr, stride: UInt32, key0: UInt32,
                                                       i0: UInt32, step: UInt32, limit: UInt32, last: UInt32 = 0):
    var w0 = uniform(i0)
    comptime if KV8:
        var kr = SIMD[U32, 4 * N](0)
        var vr = SIMD[U32, 4 * N](0)
        var ke = SIMD[DType.int32, N](0)
        var ve = SIMD[DType.int32, N](0)
        comptime for j in range(N):
            if w0 + UInt32(j) * step < limit:
                var k1 = SIMD[U32, 4](0)
                var v1 = SIMD[U32, 4](0)
                var e1 = SIMD[DType.int32, 1](0)
                var f1 = SIMD[DType.int32, 1](0)
                fetch_c[KV8, 1, 1, BOUND](k1, v1, e1, f1, kc, vc, stride, key0, i0 + UInt32(j) * step, step, last)
                kr = kr.insert[offset=4 * j](k1)
                vr = vr.insert[offset=4 * j](v1)
                ke[j] = e1[0]
                ve[j] = f1[0]
        comptime for j in range(N):
            if w0 + UInt32(j) * step < limit:
                place_c[KV8, 1, 1](t, kr.slice[4, offset=4 * j](), vr.slice[4, offset=4 * j](),
                                   SIMD[DType.int32, 1](ke[j]), SIMD[DType.int32, 1](ve[j]), i0 + UInt32(j) * step,
                                   step)
        return
    var count = 0
    comptime for j in range(N):
        if w0 + UInt32(j) * step < limit:
            count = j + 1
    comptime for n in range(N, 0, -1):
        if count == n:
            var kr = SIMD[U32, 4 * N](0)
            var vr = SIMD[U32, 4 * N](0)
            var ke = SIMD[DType.int32, N](0)
            var ve = SIMD[DType.int32, N](0)
            fetch_c[KV8, N, n, BOUND](kr, vr, ke, ve, kc, vc, stride, key0, i0, step, last)
            place_c[KV8, N, n](t, kr, vr, ke, ve, i0, step)


# A loader thread's register set S of R: one tile's N pieces of K and of V (pipelined loaders).
@always_inline
def fetch_set[KV8: Bool, N: Int, R: Int, S: Int, BOUND: Bool = False](
    mut kr: SIMD[U32, 4 * N * R],
    mut vr: SIMD[U32, 4 * N * R],
    mut ke: SIMD[DType.int32, N * R],
    mut ve: SIMD[DType.int32, N * R],
    kc: U16Ptr,
    vc: U16Ptr,
    stride: UInt32,
    key0: UInt32,
    lt: UInt32,
    last: UInt32 = 0,
):
    var k = SIMD[U32, 4 * N](0)
    var v = SIMD[U32, 4 * N](0)
    var e = SIMD[DType.int32, N](0)
    var f = SIMD[DType.int32, N](0)
    fetch_c[KV8, N, N, BOUND](k, v, e, f, kc, vc, stride, key0, lt, UInt32(NL), last)
    kr = kr.insert[offset=4 * N * S](k)
    vr = vr.insert[offset=4 * N * S](v)
    comptime if KV8:
        ke = ke.insert[offset=N * S](e)
        ve = ve.insert[offset=N * S](f)


@always_inline
def place_set[KV8: Bool, N: Int, R: Int, S: Int](
    t: SPtr,
    kr: SIMD[U32, 4 * N * R],
    vr: SIMD[U32, 4 * N * R],
    ke: SIMD[DType.int32, N * R],
    ve: SIMD[DType.int32, N * R],
    lt: UInt32,
):
    place_c[KV8, N, N](t, kr.slice[4 * N, offset=4 * N * S](), vr.slice[4 * N, offset=4 * N * S](),
                       ke.slice[N, offset=N * S](), ve.slice[N, offset=N * S](), lt, UInt32(NL))


# A tail piece: a committed key's from the cache, a path key's from the window's nodes, zeros past the row's keys.
@always_inline
def tail_piece16(
    cache: U16Ptr, nodes: U16Ptr, path: IPtr, stride: UInt32, row_stride: UInt32, p: UInt32, end: UInt32, hk: UInt32,
    key: UInt32, col: UInt32,
) -> SIMD[U32, 4]:
    # one load a lane, from its key's row: committed keys the cache's, path keys their node's, and keys past the row's
    # end its last path node's (a row that exists), zeroed (branches here cost a memory round trip a piece)
    var committed = key < p
    var slot = UInt32(0) if committed else min(key - p, end - p - 1)  # end > p: a row's path holds the row
    var row = (cache.bitcast[UInt8]() + Int((key * stride + col) * 2)) if committed else (
        nodes.bitcast[UInt8]() + Int((UInt32(path[Int(slot)]) * row_stride + hk * D + col) * 2)
    )
    var x = row.bitcast[UInt32]().load[width=4, alignment=16]()
    return x if key < end else SIMD[U32, 4](0)


@always_inline
def tail_piece8(
    cache: U16Ptr, nodes: U16Ptr, path: IPtr, stride: UInt32, row_stride: UInt32, p: UInt32, end: UInt32, hk: UInt32,
    key: UInt32, col: UInt32,
) -> SIMD[U32, 8]:
    if key < p:
        var cb = cache.bitcast[UInt8]()
        var row = key * stride
        return widen(ld4b(cb, row + col), exponent(cb, row + D))
    if key < end:
        var at = UInt32(path[Int(key - p)]) * row_stride + hk * D + col
        var out = SIMD[U32, 8](0)
        out = out.insert[offset=0](ld4(nodes, at))
        out = out.insert[offset=4](ld4(nodes, at + 8))
        return out
    return SIMD[U32, 8](0)


# This lane's query as the B operand of step t: dims 16 t + 8 half .. + 7 (zeros for no query).
@always_inline
def query(q: U16Ptr, row: UInt32, ok: Bool, half: UInt32) -> SIMD[U32, 64]:
    var qb = SIMD[U32, 64](0)
    if ok:
        comptime for t in range(16):
            qb = qb.insert[offset=4 * t](ld4(q, row * D + UInt32(16 * t) + 8 * half))
    return qb


# One 16-key tile into a wave's 16 queries (attention_rocm.cu's ``fold``), keys outside ``valid`` (bit j: key j) at
# -inf. A query's scores sit in lanes c and c + 16 (keys 8 half + i); O row 8 half + i is o[8 n + i], n = dim tile.
@always_inline
def fold(
    t: SPtr,
    qb: SIMD[U32, 64],
    mut o: SIMD[F32, 128],
    mut m: Float32,
    mut l: Float32,
    valid: UInt32,
    scale: Float32,
    c: UInt32,
    half: UInt32,
):
    var NEG = neg_inf()
    var ka = t + Int((c * KROW + 8 * half) >> 1)  # this lane's K row; a step's operand at a constant offset
    var vb = t + Int((VT + c * 16 + 8 * half) >> 1)  # this lane's V row of dim tile 0
    var s = SIMD[F32, 8](0)
    comptime for d in range(16):
        var a = (ka + 8 * d).load[width=4, alignment=16]()
        s = wmma(a, qb.slice[4, offset=4 * d](), s)
    var mt = NEG
    comptime for i in range(8):
        var si = s[i] * scale if ((valid >> (8 * half + UInt32(i))) & 1) != 0 else NEG
        s[i] = si
        mt = maxnum(mt, si)
    var lane = lane_id()
    mt = maxnum(mt, shfl(mt, lane ^ 16))
    var active = mt != NEG
    var next = maxnum(m, mt) if active else m
    var alpha = (Float32(0.0) if m == NEG else fast_exp(m - next)) if active else Float32(1.0)
    var p = SIMD[F32, 8](0)
    var sum = Float32(0.0)
    comptime for i in range(8):
        p[i] = fast_exp(s[i] - next) if active and s[i] != NEG else Float32(0.0)
        sum = sum + p[i]
    sum = sum + shfl(sum, lane ^ 16)  # the query's two halves (addition commutes)
    l = l.fma(alpha, sum)
    m = next
    if ballot(alpha != Float32(1.0)) != 0:  # O rows 8 half + i: rescale those whose factor moved
        comptime for i in range(8):
            var a = shfl(alpha, 8 * half + UInt32(i))
            if a != Float32(1.0):
                comptime for n in range(16):
                    o[8 * n + i] = o[8 * n + i] * a
    var pa = SIMD[U32, 4](pack2(p[0], p[1]), pack2(p[2], p[3]), pack2(p[4], p[5]), pack2(p[6], p[7]))
    comptime for n in range(16):
        var b = (vb + 128 * n).load[width=4, alignment=16]()
        o = o.insert[offset=8 * n](wmma(pa, b, o.slice[8, offset=8 * n]()))


# Item (stream, first pair, chunk) of full committed chunks, grid (KV heads, items): a block of CW compute waves takes
# the items whose first pair starts a run of CW tiles (16 CW pairs), the rest return. Pair r of a stream is row r / G,
# head hk G + r % G. PIPE: four more waves load through a double buffer (tile kt in buffer kt % 2 while tile kt + 1 is
# placed and kt + 3 fetched); else every wave loads each tile, then the compute waves fold it. KV8: packed FP8 rows.
@always_inline
def shared_body[PIPE: Bool, KV8: Bool](
    q: U16Ptr,
    base: U16Ptr,
    offs: LPtr,
    streams: IPtr,
    items: IPtr,
    po: FPtr,
    pm: FPtr,
    pl: FPtr,
    w: Int32,
    h: Int32,
    hk_count: Int32,
    g32: Int32,
    scale: Float32,
    item0: Int32,
):
    comptime NT = 2 if PIPE else 1
    var tb = stack_allocation[NT * TILE, UInt32, alignment=16, address_space=AddressSpace.SHARED]()
    var tb1 = tb + (TILE if PIPE else 0)
    var tid = UInt32(thread_idx.x)
    var bd = UInt32(block_dim.x)
    var cw = (bd >> 5) - (UInt32(LOADERS) if PIPE else 0)
    var item = UInt32(item0) + UInt32(block_idx.y)
    var hk = UInt32(block_idx.x)
    var g = UInt32(g32)
    var s = UInt32(items[Int(3 * item)])
    var first = UInt32(items[Int(3 * item + 1)])
    var chunk = UInt32(items[Int(3 * item + 2)])
    var start = UInt32(streams[Int(4 * s)])
    var rows = UInt32(streams[Int(4 * s + 1)])
    var p = UInt32(streams[Int(4 * s + 2)])
    if first % (16 * cw) != 0 or (chunk + 1) * CH > p:  # another block's tiles; a padded plan's missing chunk
        return
    var lane = tid & 31
    var wave = tid >> 5
    var c = lane & 15
    var half = lane >> 4
    var loader = wave >= cw
    var pairs = rows * g
    var first_pair = first + 16 * wave
    var live = not loader and first_pair < pairs  # wave-uniform
    var kc: U16Ptr
    var vc: U16Ptr
    var stride: UInt32
    comptime if KV8:
        var b8 = base.bitcast[UInt8]()
        kc = (b8 + Int(offs[Int(2 * s)]) + Int(hk * ROW8)).bitcast[UInt16]()
        vc = (b8 + Int(offs[Int(2 * s + 1)]) + Int(hk * ROW8)).bitcast[UInt16]()
        stride = UInt32(hk_count) * ROW8
    else:
        kc = base + Int(offs[Int(2 * s)]) + Int(hk * D)
        vc = base + Int(offs[Int(2 * s + 1)]) + Int(hk * D)
        stride = UInt32(hk_count) * D
    var key0 = chunk * CH
    comptime P = P8 if KV8 else P16
    comptime if PIPE:
        if loader:  # attention_rocm.cu's load_tiles over the chunk's 32 tiles, DEPTH tiles in registers
            comptime N = P // NL
            comptime R = DEPTH
            comptime nt = CH // 16
            var lt = tid - 32 * cw
            var kr = SIMD[U32, 4 * N * R](0)
            var vr = SIMD[U32, 4 * N * R](0)
            var ke = SIMD[DType.int32, N * R](0)
            var ve = SIMD[DType.int32, N * R](0)
            comptime for r in range(R):
                fetch_set[KV8, N, R, r](kr, vr, ke, ve, kc, vc, stride, key0 + UInt32(16 * r), lt)
            place_set[KV8, N, R, 0](tb, kr, vr, ke, ve, lt)
            fetch_set[KV8, N, R, 0](kr, vr, ke, ve, kc, vc, stride, key0 + UInt32(16 * R), lt)
            barrier()
            # step kt (the compute waves fold tile kt): place tile kt + 1 from its registers into the other buffer,
            # fetch tile kt + 1 + R into them; every step a barrier
            var kt = UInt32(0)
            while kt + UInt32(2 * R) < UInt32(nt):  # groups of R steps whose fetches all fall in the chunk
                comptime for u in range(R):
                    place_set[KV8, N, R, (u + 1) % R](tb1 if (u + 1) % 2 == 1 else tb, kr, vr, ke, ve, lt)
                    fetch_set[KV8, N, R, (u + 1) % R](kr, vr, ke, ve, kc, vc, stride,
                                                      key0 + 16 * (kt + UInt32(u + 1 + R)), lt)
                    barrier()
                kt += UInt32(R)
            comptime for u in range(2 * R):  # the last 2R steps (kt = nt - 2R + u): fetch and place what is left
                comptime if nt - 2 * R + u + 1 < nt:
                    place_set[KV8, N, R, (u + 1) % R](tb1 if (u + 1) % 2 == 1 else tb, kr, vr, ke, ve, lt)
                comptime if nt - 2 * R + u + 1 + R < nt:
                    fetch_set[KV8, N, R, (u + 1) % R](kr, vr, ke, ve, kc, vc, stride,
                                                      key0 + UInt32(16 * (nt - 2 * R + u + 1 + R)), lt)
                barrier()
            return
    var mine = first_pair + c
    var qb = query(q, (start + mine // g) * UInt32(h) + hk * g + mine % g, live and mine < pairs, half)
    var o = SIMD[F32, 128](0)
    var m = neg_inf()
    var l = Float32(0.0)
    comptime if PIPE:
        barrier()
        var kt = 0
        while kt < CH // 16:
            if live:
                fold(tb, qb, o, m, l, 0xFFFF, scale, c, half)
            barrier()
            if live:
                fold(tb1, qb, o, m, l, 0xFFFF, scale, c, half)
            barrier()
            kt += 2
    else:
        comptime N = 4 if not KV8 else 2  # attention_rocm.cu's Stage16::B, Stage8::B
        for kt in range(CH // 16):
            barrier()  # the previous tile is consumed
            var k0 = key0 + UInt32(16 * kt)
            var b = UInt32(0)
            while b < UInt32(P):
                stage_batch[KV8, N](tb, kc, vc, stride, k0, b + tid, bd, UInt32(P))
                b += UInt32(N) * bd
            barrier()
            if live:
                fold(tb, qb, o, m, l, 0xFFFF, scale, c, half)
    if live:
        comptime for i in range(8):
            var r = first_pair + 8 * half + UInt32(i)
            if r < pairs:
                var at = ((Int(chunk) * Int(w) + Int(start + r // g)) * Int(h) + Int(hk * g + r % g)) * D
                comptime for n in range(16):
                    po[at + 16 * n + Int(c)] = o[8 * n + i]
        if half == 0 and mine < pairs:
            var at = (Int(chunk) * Int(w) + Int(start + mine // g)) * Int(h) + Int(hk * g + mine % g)
            pm[at] = m
            pl[at] = l


# One tail tile into LDS by the four loader waves (lt: 0..127): every piece loaded, then put.
@always_inline
def tail_tile[KV8: Bool](
    t: SPtr, kc: U16Ptr, vc: U16Ptr, kn: U16Ptr, vn: U16Ptr, path: IPtr, stride: UInt32, nstride: UInt32,
    vstride: UInt32, p: UInt32, end: UInt32, hk: UInt32, k0: UInt32, lt: UInt32,
):
    comptime if KV8:
        comptime N = P8 // NL
        var kr = SIMD[U32, 8 * N](0)
        var vr = SIMD[U32, 8 * N](0)
        comptime for j in range(N):
            var i = lt + UInt32(j * NL)
            kr = kr.insert[offset=8 * j](tail_piece8(kc, kn, path, stride, nstride, p, end, hk, k0 + (i >> 4),
                                                     (i & 15) * 16))
            vr = vr.insert[offset=8 * j](tail_piece8(vc, vn, path, stride, vstride, p, end, hk, k0 + (i & 15),
                                                     (i >> 4) * 16))
        comptime for j in range(N):
            var i = lt + UInt32(j * NL)
            put_k16(t, i, kr.slice[8, offset=8 * j]())
            put_v16(t, i, vr.slice[8, offset=8 * j]())
    else:
        comptime N = P16 // NL
        var kr = SIMD[U32, 4 * N](0)
        var vr = SIMD[U32, 4 * N](0)
        comptime for j in range(N):
            var i = lt + UInt32(j * NL)
            kr = kr.insert[offset=4 * j](tail_piece16(kc, kn, path, stride, nstride, p, end, hk, k0 + (i >> 5),
                                                      (i & 31) * 8))
            vr = vr.insert[offset=4 * j](tail_piece16(vc, vn, path, stride, vstride, p, end, hk, k0 + (i & 15),
                                                      (i >> 4) * 8))
        comptime for j in range(N):
            var i = lt + UInt32(j * NL)
            put_k(t, i, kr.slice[4, offset=4 * j]())
            put_v(t, i, vr.slice[4, offset=4 * j]())


# Row, KV head, tail chunk (grid (W, KV heads, tails)): keys from the chunk's start to the last committed one from
# the cache, then the row's path from the window's own keys and values. Four loader waves fill a double buffer, a
# fifth wave folds (the row's G heads are its queries); attention_rocm.cu's wave 0 loads and folds, the same tiles.
@always_inline
def tail_body[KV8: Bool](
    q: U16Ptr,
    kn: U16Ptr,
    vn: U16Ptr,
    base: U16Ptr,
    offs: LPtr,
    streams: IPtr,
    row_stream: IPtr,
    paths: IPtr,
    depths: IPtr,
    po: FPtr,
    pm: FPtr,
    pl: FPtr,
    w: Int32,
    vs: Int32,
    h: Int32,
    hk_count: Int32,
    g32: Int32,
    scale: Float32,
):
    var t = stack_allocation[2 * TILE, UInt32, alignment=16, address_space=AddressSpace.SHARED]()
    var t1 = t + TILE
    var node = UInt32(block_idx.x)
    var hk = UInt32(block_idx.y)
    var g = UInt32(g32)
    var s = UInt32(row_stream[Int(node)])
    var p = UInt32(streams[Int(4 * s + 2)])
    var nch = UInt32(streams[Int(4 * s + 3)])
    var chunk = p // CH + UInt32(block_idx.z)
    if chunk >= nch:
        return
    var tid = UInt32(thread_idx.x)
    var lane = tid & 31
    var c = lane & 15
    var half = lane >> 4
    var folds = tid >= NL  # wave 4 folds, waves 0-3 load
    var end = p + UInt32(depths[Int(node)])  # keys [0, p) committed, [p, end) the row's path
    var key0 = chunk * CH
    var nt = min(UInt32(CH // 16), (end - key0 + 15) // 16) if end > key0 else UInt32(0)
    var kc: U16Ptr
    var vc: U16Ptr
    var stride: UInt32
    comptime if KV8:
        var b8 = base.bitcast[UInt8]()
        kc = (b8 + Int(offs[Int(2 * s)]) + Int(hk * ROW8)).bitcast[UInt16]()
        vc = (b8 + Int(offs[Int(2 * s + 1)]) + Int(hk * ROW8)).bitcast[UInt16]()
        stride = UInt32(hk_count) * ROW8
    else:
        kc = base + Int(offs[Int(2 * s)]) + Int(hk * D)
        vc = base + Int(offs[Int(2 * s + 1)]) + Int(hk * D)
        stride = UInt32(hk_count) * D
    var path = paths + Int(node) * MAXD
    var nstride = UInt32(hk_count) * D
    var vstride = UInt32(vs)
    if not folds:  # four loader waves: tile kt + 1 into the other buffer while tile kt folds
        # keys counted from the tail's first chunk (``from_key``: the committed keys a tail reads sit in that chunk,
        # so their 32-bit offsets never wrap); key - p and end - p, all a piece compares, are the same counted so
        var first = p // CH * CH
        var kf = from_key[KV8](kc, first, stride)
        var vf = from_key[KV8](vc, first, stride)
        var pf = p - first
        var ef = end - first
        var k0f = key0 - first
        if nt > 0:
            tail_tile[KV8](t, kf, vf, kn, vn, path, stride, nstride, vstride, pf, ef, hk, k0f, tid)
        barrier()
        for kt in range(Int(nt)):
            if UInt32(kt) + 1 < nt:
                tail_tile[KV8](t1 if kt % 2 == 0 else t, kf, vf, kn, vn, path, stride, nstride, vstride, pf, ef, hk,
                               k0f + 16 * UInt32(kt + 1), tid)
            barrier()
        return
    var qb = query(q, node * UInt32(h) + hk * g + c, c < g, half)
    var o = SIMD[F32, 128](0)
    var m = neg_inf()
    var l = Float32(0.0)
    barrier()
    for kt in range(Int(nt)):
        var k0 = key0 + 16 * UInt32(kt)
        var valid = UInt32(0xFFFF) if k0 + 16 <= end else (UInt32(1) << (end - k0)) - 1
        fold(t if kt % 2 == 0 else t1, qb, o, m, l, valid, scale, c, half)
        barrier()
    if folds:
        comptime for i in range(8):
            var x = 8 * half + UInt32(i)
            if x < g:
                var at = ((Int(chunk) * Int(w) + Int(node)) * Int(h) + Int(hk * g + x)) * D
                comptime for n in range(16):
                    po[at + 16 * n + Int(c)] = o[8 * n + i]
        if half == 0 and c < g:
            var at = (Int(chunk) * Int(w) + Int(node)) * Int(h) + Int(hk * g + c)
            pm[at] = m
            pl[at] = l


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](384))
def shared_pipe(
    q: U16Ptr, base: U16Ptr, offs: LPtr, streams: IPtr, items: IPtr, po: FPtr, pm: FPtr, pl: FPtr,
    w: Int32, h: Int32, hk_count: Int32, g: Int32, scale: Float32, item0: Int32,
):
    shared_body[True, False](q, base, offs, streams, items, po, pm, pl, w, h, hk_count, g, scale, item0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](384))
def shared_pipe8(
    q: U16Ptr, base: U16Ptr, offs: LPtr, streams: IPtr, items: IPtr, po: FPtr, pm: FPtr, pl: FPtr,
    w: Int32, h: Int32, hk_count: Int32, g: Int32, scale: Float32, item0: Int32,
):
    shared_body[True, True](q, base, offs, streams, items, po, pm, pl, w, h, hk_count, g, scale, item0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def shared_flat(
    q: U16Ptr, base: U16Ptr, offs: LPtr, streams: IPtr, items: IPtr, po: FPtr, pm: FPtr, pl: FPtr,
    w: Int32, h: Int32, hk_count: Int32, g: Int32, scale: Float32, item0: Int32,
):
    shared_body[False, False](q, base, offs, streams, items, po, pm, pl, w, h, hk_count, g, scale, item0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def shared_flat8(
    q: U16Ptr, base: U16Ptr, offs: LPtr, streams: IPtr, items: IPtr, po: FPtr, pm: FPtr, pl: FPtr,
    w: Int32, h: Int32, hk_count: Int32, g: Int32, scale: Float32, item0: Int32,
):
    shared_body[False, True](q, base, offs, streams, items, po, pm, pl, w, h, hk_count, g, scale, item0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NL + 32))
def tail16(
    q: U16Ptr, kn: U16Ptr, vn: U16Ptr, base: U16Ptr, offs: LPtr, streams: IPtr, row_stream: IPtr, paths: IPtr,
    depths: IPtr, po: FPtr, pm: FPtr, pl: FPtr, w: Int32, vs: Int32, h: Int32, hk_count: Int32, g: Int32,
    scale: Float32,
):
    tail_body[False](q, kn, vn, base, offs, streams, row_stream, paths, depths, po, pm, pl, w, vs, h, hk_count, g,
                     scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NL + 32))
def tail8(
    q: U16Ptr, kn: U16Ptr, vn: U16Ptr, base: U16Ptr, offs: LPtr, streams: IPtr, row_stream: IPtr, paths: IPtr,
    depths: IPtr, po: FPtr, pm: FPtr, pl: FPtr, w: Int32, vs: Int32, h: Int32, hk_count: Int32, g: Int32,
    scale: Float32,
):
    tail_body[True](q, kn, vn, base, offs, streams, row_stream, paths, depths, po, pm, pl, w, vs, h, hk_count, g,
                    scale)


# attention.py's Triton _merge: a row's chunks in key order, each output column's fold its own and every column of a
# head recomputing m and l alike. A block takes MERGE_HEADS (row, head) pairs, 64 threads a pair, 4 columns a thread
# (grid (W, ceil(H / MERGE_HEADS))); the arithmetic per element is _merge's as Triton compiles it.
# CO_FIRST: Triton contracts the column fold o * a + co * b as fma(o, a, b * co) where G < 16 masks the head rows, and
# as fma(co, b, o * a) at G = 16 (no mask: the backend schedules the multiplies the other way); l is fma(cl, b, l * a)
# in both. ``merge16`` is the G = 16 form.
@always_inline
def merge_body[CO_FIRST: Bool](
    po: FPtr, pm: FPtr, pl: FPtr, dst: U16Ptr, streams: IPtr, row_stream: IPtr, w: Int32, h: Int32
):
    var node = UInt32(block_idx.x)
    var tid = UInt32(thread_idx.x)
    var head = UInt32(block_idx.y) * MERGE_HEADS + (tid >> 6)
    var col = (tid & 63) * 4
    if head >= UInt32(h):
        return
    var nch = UInt32(streams[Int(4 * UInt32(row_stream[Int(node)]) + 3)])
    var NEG = neg_inf()
    var m = NEG
    var l = Float32(0.0)
    var o = SIMD[F32, 4](0)
    var pair = Int(node) * Int(h) + Int(head)
    var step = Int(w) * Int(h)
    for chunk in range(Int(nch)):
        var at = chunk * step + pair
        var cm = pm[at]
        var cl = pl[at]
        var co = (po + at * D + Int(col)).load[width=4, alignment=16]()
        var active = cl > Float32(0.0)
        var next = maxnum(m, cm) if active else m
        var a = (Float32(0.0) if m == NEG else triton_exp(m - next)) if active else Float32(1.0)
        var b = triton_exp(cm - next) if active else Float32(0.0)
        comptime for i in range(4):
            comptime if CO_FIRST:
                o[i] = co[i].fma(b, o[i] * a)
            else:
                o[i] = o[i].fma(a, b * co[i])
        l = cl.fma(b, l * a)
        m = next
    var r = SIMD[U32, 4](0)
    comptime for i in range(4):
        var x = o[i] / l
        r[i] = UInt32(0x7FFF) if x != x else bf16_round(x)
    var at = (Int(node) * Int(h) + Int(head)) * D + Int(col)
    var packed = SIMD[U32, 2](r[0] | (r[1] << 16), r[2] | (r[3] << 16))
    (dst + at).bitcast[UInt32]().store[alignment=8](packed)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](64 * MERGE_HEADS))
def merge(po: FPtr, pm: FPtr, pl: FPtr, dst: U16Ptr, streams: IPtr, row_stream: IPtr, w: Int32, h: Int32):
    merge_body[False](po, pm, pl, dst, streams, row_stream, w, h)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](64 * MERGE_HEADS))
def merge16(po: FPtr, pm: FPtr, pl: FPtr, dst: U16Ptr, streams: IPtr, row_stream: IPtr, w: Int32, h: Int32):
    merge_body[True](po, pm, pl, dst, streams, row_stream, w, h)


# Prompt attention (attention_rocm.cu's prompt_kernel): q (W, H, D), caches (T, HK, D) bf16 or (T, HK, ROW8) packed
# holding keys [0, p0 + W), dst (W, H, D). A block per 16 rb query rows and KV head (the longest causal blocks
# first), a compute wave per 16 rows and query head (g heads a KV head: g rb compute waves); PIPE: four loader waves
# fill a double buffer (attention_rocm.cu's load_tiles: two tiles in registers, one barrier a tile), else every wave
# stages each tile. A row's keys stop at its own position; none of these change a row's bits.
@always_inline
def prompt_body[PIPE: Bool, KV8: Bool](
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0_32: Int32, w32: Int32, h32: Int32, hk_count: Int32,
    g32: Int32, rb32: Int32, scale: Float32,
):
    comptime NT = 2 if PIPE else 1
    var tb = stack_allocation[NT * TILE, UInt32, alignment=16, address_space=AddressSpace.SHARED]()
    var tb1 = tb + (TILE if PIPE else 0)
    var tid = UInt32(thread_idx.x)
    var bd = UInt32(block_dim.x)
    var p0 = UInt32(p0_32)
    var w = UInt32(w32)
    var h = UInt32(h32)
    var g = UInt32(g32)
    var rb = UInt32(rb32)
    var cw = g * rb
    var r0 = (UInt32(grid_dim.x) - 1 - UInt32(block_idx.x)) * 16 * rb
    var kvh = UInt32(block_idx.y)
    var wave = tid >> 5
    var nt = (p0 + min(r0 + 16 * rb, w) - 1) // 16 + 1
    var last = p0 + w - 1  # the last key
    var kb: U16Ptr
    var vb: U16Ptr
    var stride: UInt32
    comptime if KV8:
        kb = (kc.bitcast[UInt8]() + Int(kvh * ROW8)).bitcast[UInt16]()
        vb = (vc.bitcast[UInt8]() + Int(kvh * ROW8)).bitcast[UInt16]()
        stride = UInt32(hk_count) * ROW8
    else:
        kb = kc + Int(kvh * D)
        vb = vc + Int(kvh * D)
        stride = UInt32(hk_count) * D
    comptime P = P8 if KV8 else P16
    comptime if PIPE:
        if wave >= cw:  # load_tiles: tile kt in buffer kt % 2 while tile kt + 1 is placed and kt + 3 fetched
            comptime N = P // NL
            var lt = tid - 32 * cw
            var kr = SIMD[U32, 8 * N](0)
            var vr = SIMD[U32, 8 * N](0)
            var ke = SIMD[DType.int32, 2 * N](0)
            var ve = SIMD[DType.int32, 2 * N](0)
            fetch_set[KV8, N, 2, 0, True](kr, vr, ke, ve, kb, vb, stride, 0, lt, last)
            if nt > 1:
                fetch_set[KV8, N, 2, 1, True](kr, vr, ke, ve, kb, vb, stride, 16, lt, last)
            place_set[KV8, N, 2, 0](tb, kr, vr, ke, ve, lt)
            if nt > 2:
                fetch_set[KV8, N, 2, 0, True](kr, vr, ke, ve, kb, vb, stride, 32, lt, last)
            barrier()
            var kt = UInt32(0)
            while kt < nt:
                if kt + 1 < nt:
                    place_set[KV8, N, 2, 1](tb1, kr, vr, ke, ve, lt)
                    if kt + 3 < nt:
                        fetch_set[KV8, N, 2, 1, True](kr, vr, ke, ve, kb, vb, stride, 16 * (kt + 3), lt, last)
                barrier()
                if kt + 1 >= nt:
                    break
                if kt + 2 < nt:
                    place_set[KV8, N, 2, 0](tb, kr, vr, ke, ve, lt)
                    if kt + 4 < nt:
                        fetch_set[KV8, N, 2, 0, True](kr, vr, ke, ve, kb, vb, stride, 16 * (kt + 4), lt, last)
                barrier()
                kt += 2
            return
    var lane = tid & 31
    var c = lane & 15
    var half = lane >> 4
    var rows0 = r0 + 16 * (wave // g)
    var head = kvh * g + wave % g
    var live = rows0 < w  # wave-uniform
    var row = rows0 + c
    var pos = p0 + row
    var qb = query(q, min(row, w - 1) * h + head, live, half)
    var o = SIMD[F32, 128](0)
    var m = neg_inf()
    var l = Float32(0.0)
    comptime if PIPE:
        barrier()
        var kt = UInt32(0)
        while kt < nt:
            if live:
                fold(tb, qb, o, m, l, prompt_valid(row, w, pos, 16 * kt), scale, c, half)
            barrier()
            if kt + 1 >= nt:
                break
            if live:
                fold(tb1, qb, o, m, l, prompt_valid(row, w, pos, 16 * (kt + 1)), scale, c, half)
            barrier()
            kt += 2
    else:
        comptime N = 4 if not KV8 else 2  # attention_rocm.cu's Stage16::B, Stage8::B
        for kt in range(Int(nt)):
            barrier()  # the previous tile is consumed
            var k0 = UInt32(16 * kt)
            var b = UInt32(0)
            while b < UInt32(P):
                stage_batch[KV8, N, True](tb, kb, vb, stride, k0, b + tid, bd, UInt32(P), last)
                b += UInt32(N) * bd
            barrier()
            if live:
                fold(tb, qb, o, m, l, prompt_valid(row, w, pos, k0), scale, c, half)
    if not live:
        return
    comptime for i in range(8):
        var li = shfl(l, 8 * half + UInt32(i))
        var r = rows0 + 8 * half + UInt32(i)
        if r < w:
            var at = (Int(r) * Int(h) + Int(head)) * D + Int(c)
            comptime for n in range(16):
                dst[at + 16 * n] = UInt16(bf16_round(o[8 * n + i] / li))


# Keys of tile ``key0`` at or before this lane's row (bit j: key key0 + j), none for rows past W.
@always_inline
def prompt_valid(row: UInt32, w: UInt32, pos: UInt32, key0: UInt32) -> UInt32:
    if row >= w or pos < key0:
        return 0
    return UInt32(0xFFFF) if pos >= key0 + 15 else (UInt32(2) << (pos - key0)) - 1


# prompt_{p: loaders, f: none}{waves a block}{b: bf16, k: packed caches}: one code object a block size (8, 12, 16 or
# 20 waves), the launcher taking the smallest that holds its block. A kernel bounded at the block it runs schedules
# for that occupancy (bounded at 640 threads, a 512-thread block measured 5-8% slower). Names stay short: longer ones
# lose the separator in the symbol cuda/mojo.py matches.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def prompt_p8b(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, False](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def prompt_p8k(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, True](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](384))
def prompt_p12b(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, False](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](384))
def prompt_p12k(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, True](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](512))
def prompt_p16b(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, False](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](512))
def prompt_p16k(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, True](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](640))
def prompt_p20b(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, False](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](640))
def prompt_p20k(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[True, True](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def prompt_f8b(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[False, False](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def prompt_f8k(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[False, True](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](384))
def prompt_f12b(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[False, False](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](384))
def prompt_f12k(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[False, True](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](512))
def prompt_f16b(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[False, False](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](512))
def prompt_f16k(
    q: U16Ptr, kc: U16Ptr, vc: U16Ptr, dst: U16Ptr, p0: Int32, w: Int32, h: Int32, hk_count: Int32, g: Int32,
    rb: Int32, scale: Float32,
):
    prompt_body[False, True](q, kc, vc, dst, p0, w, h, hk_count, g, rb, scale)


@export
def tf_attention_instantiate(a: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var u = U16Ptr(unsafe_from_address=a)
        var f = FPtr(unsafe_from_address=a)
        var i = IPtr(unsafe_from_address=a)
        var o = LPtr(unsafe_from_address=a)
        var z = Int32(0)
        var x = Float32(0)
        ctx.enqueue_function[shared_pipe](u, u, o, i, i, f, f, f, z, z, z, z, x, z, grid_dim=1, block_dim=32)
        ctx.enqueue_function[shared_pipe8](u, u, o, i, i, f, f, f, z, z, z, z, x, z, grid_dim=1, block_dim=32)
        ctx.enqueue_function[shared_flat](u, u, o, i, i, f, f, f, z, z, z, z, x, z, grid_dim=1, block_dim=32)
        ctx.enqueue_function[shared_flat8](u, u, o, i, i, f, f, f, z, z, z, z, x, z, grid_dim=1, block_dim=32)
        ctx.enqueue_function[tail16](u, u, u, u, o, i, i, i, i, f, f, f, z, z, z, z, z, x, grid_dim=1, block_dim=NL + 32)
        ctx.enqueue_function[tail8](u, u, u, u, o, i, i, i, i, f, f, f, z, z, z, z, z, x, grid_dim=1, block_dim=NL + 32)
        ctx.enqueue_function[merge](f, f, f, u, i, i, z, z, grid_dim=1, block_dim=64 * MERGE_HEADS)
        ctx.enqueue_function[merge16](f, f, f, u, i, i, z, z, grid_dim=1, block_dim=64 * MERGE_HEADS)
        ctx.enqueue_function[prompt_p8b](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_p8k](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_p12b](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_p12k](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_p16b](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_p16k](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_p20b](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_p20k](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_f8b](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_f8k](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_f12b](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_f12k](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_f16b](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        ctx.enqueue_function[prompt_f16k](u, u, u, u, z, z, z, z, z, z, x, grid_dim=1, block_dim=32)
        return 0
    except:
        return 1
