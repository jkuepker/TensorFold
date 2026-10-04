# The ROCm decode lane matmul (qmm_rocm.cu's wmma_kernel and reduce_kernel) in Mojo, for gfx12 (wave32), launched
# from qmm_rocm_mojo.cu with hipModuleLaunchKernel (TF_ROCM_LANE=mojo). The arithmetic is the HIP kernel's, step for
# step, so the two lanes give the same bits: each 64-input group's dot on bf16 128 + q from four chained
# v_wmma_f32_16x16x16_bf16 (K steps 0..3, zero start), then acc = fma(xs, b - 128 s, fma(p, s, acc)) in group order,
# K slices fixed by the weight's shape added in slice order, bf16 rounded to nearest even by the same integer formula.
#
# Kernels are top-level defs (one per instantiation: mojo2hsaco names a code object after its def) and must be
# instantiated by a host function, which is what tf_qmm_instantiate is for; it is never called.
from std.sys import llvm_intrinsic
from std.memory import bitcast, stack_allocation
from std.atomic import Atomic, Ordering, fence
from std.utils import StaticTuple
from max.gpu import thread_idx, block_idx, barrier, MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext

comptime F32 = DType.float32
comptime U32 = DType.uint32
comptime U16 = DType.uint16
comptime U32Ptr = UnsafePointer[UInt32, MutAnyOrigin]
comptime U16Ptr = UnsafePointer[UInt16, MutAnyOrigin]
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime IPtr = UnsafePointer[Int32, MutAnyOrigin]

comptime WARPS = 8  # a block: 8 warps x 16 outputs
comptime THREADS = 32 * WARPS
comptime COLS = 16 * WARPS
comptime SLAB = 4  # groups a slab stages (their WMMA chains interleave)
comptime XROW = SLAB * 8 + 1  # uint4 a staged row: 8 bf16 a uint4; +1: rows off each other's banks


@always_inline
def bf16_value(v: UInt32) -> Float32:
    return bitcast[F32, 1](v << 16)


@always_inline
def bf16_round(f: Float32) -> UInt16:
    var u = bitcast[U32, 1](f)
    return UInt16((u + 0x7FFF + ((u >> 16) & 1)) >> 16)


# Inputs 2j and 2j + 1 of a word as the bf16 pair (128 + q, 128 + q), for j = 0..3.
@always_inline
def pairs(w: UInt32) -> SIMD[U32, 4]:
    var out = SIMD[U32, 4](0)
    comptime for j in range(4):
        out[j] = ((w >> UInt32(4 * j)) & 0x000F000F) | 0x43004300
    return out


# Tiled layout (qmm_groups.py): output col of group g at ((col / 16) * kg + g) * 16 + col % 16 (64-bit, as the HIP
# kernel's size_t).
@always_inline
def tile_at(g: UInt32, kg: UInt32, col: UInt32) -> Int:
    return (Int(col >> 4) * Int(kg) + Int(g)) * 16 + Int(col & 15)


# A slab's reads for a lane: per group j its 16 bytes of words (nontemporal: read once a step, keep L2 for x and
# partials), the output's scale and bias; zeros past the slice or N. Scales and biases each take a 32-bit lane: packed
# two to a register, the second half's load waits out the first's (a full memory round trip inside the prefetch).
@always_inline
def slab(
    mut w: SIMD[U32, 4 * SLAB],
    mut s: SIMD[U32, SLAB],
    mut b: SIMD[U32, SLAB],
    words: U32Ptr,
    scales: U16Ptr,
    biases: U16Ptr,
    it: UInt32,
    gs: UInt32,
    items: UInt32,
    blocks: UInt32,
    gps: UInt32,
    kg: UInt32,
    n: UInt32,
    lc: UInt32,
    h: UInt32,
):
    var col = (it % blocks) * COLS + lc
    var g1 = min(kg, (it // blocks) * gps + gps)
    var ok = it < items and col < n
    comptime for j in range(SLAB):
        var v = SIMD[U32, 4](0)
        var sv = UInt32(0)
        var bv = UInt32(0)
        if ok and gs + UInt32(j) < g1:
            var at = tile_at(gs + UInt32(j), kg, col)
            v = (words + (at * 2 + Int(h)) * 4).load[width=4, alignment=16, non_temporal=True]()
            sv = UInt32(scales[at])
            bv = UInt32(biases[at])
        comptime for k in range(4):
            w[4 * j + k] = v[k]
        s[j] = sv
        b[j] = bv


# A slab's input rows (bf16 pairs, uint4 at a time) and group sums for this thread to stage; zeros past m or the slice.
@always_inline
def stage[MT: Int](
    mut xv: SIMD[U32, 4 * (MT * 16 * SLAB * 8 // THREADS)],
    mut sv: Float32,
    x: U32Ptr,
    xs: FPtr,
    ld: UInt32,
    m: UInt32,
    kg: UInt32,
    gs: UInt32,
    groups: UInt32,
    tid: UInt32,
):
    xv = SIMD[U32, 4 * (MT * 16 * SLAB * 8 // THREADS)](0)
    comptime for q in range(MT * 16 * SLAB * 8 // THREADS):
        var i = tid + UInt32(q * THREADS)
        var r = i // (SLAB * 8)
        var cc = i % (SLAB * 8)
        if r < m and cc < groups * 8:
            xv = xv.insert[offset=4 * q]((x + Int(r) * Int(ld) + Int(gs * 32 + cc * 4)).load[width=4, alignment=16]())
    sv = Float32(0)
    var sr = tid // SLAB
    var scc = tid % SLAB
    if tid < UInt32(MT * 16 * SLAB) and sr < m and scc < groups:
        sv = xs[Int(sr) * Int(kg) + Int(gs + scc)]


# An item is 128 outputs (8 warps x 16) over one K slice, walked in slabs of SLAB groups whose 16 * MT input rows sit
# in shared memory. Blocks are persistent: block i takes items i, i + nblocks, ..., its weight reads a slab ahead
# across items. Lane l: B and D column l % 16; half h = l / 16 holds a group's inputs 32h..32h+31, K step s taking
# inputs 32h + 8s .. 32h + 8s + 7 (word 4h + s) for B and for A's row l % 16. D holds rows 8h .. 8h + 7.
# Split K: an item stores its slice's sums; the one that finds its column's other slices done adds them in slice
# order and writes the output; counts[column block] returns to zero for the next call. Output and partial-sum
# indices fit 32 bits (at most 16 slices x 32 rows x N); 64-bit ones, hoisted out of the loop, cost registers.
@always_inline
def wmma_body[MT: Int](
    x: U32Ptr,
    ldx2: Int32,
    m32: Int32,
    xs: FPtr,
    kg32: Int32,
    gps32: Int32,
    slices32: Int32,
    words: U32Ptr,
    scales: U16Ptr,
    biases: U16Ptr,
    n32: Int32,
    part: FPtr,
    dst: U16Ptr,
    f32: Int32,
    counts: IPtr,
    nblocks32: Int32,
):
    var xsh = stack_allocation[MT * 16 * XROW * 4, UInt32, alignment=16, address_space=AddressSpace.SHARED]()
    var xssh = stack_allocation[MT * 16 * SLAB, Float32, alignment=16, address_space=AddressSpace.SHARED]()
    var last = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()
    var m = UInt32(m32)
    var kg = UInt32(kg32)
    var gps = UInt32(gps32)
    var slices = UInt32(slices32)
    var n = UInt32(n32)
    var nblocks = UInt32(nblocks32)
    var ld = UInt32(ldx2)
    var blocks = (n + COLS - 1) // COLS
    var items = blocks * slices
    var tid = UInt32(thread_idx.x)
    var lane = tid & 31
    var warp = tid >> 5
    var h = lane >> 4
    var c = lane & 15
    var lc = warp * 16 + c  # the lane's output within an item
    var item = UInt32(block_idx.x)
    if item >= items:
        return
    var gs = (item // blocks) * gps
    var cw = SIMD[U32, 4 * SLAB](0)
    var cs = SIMD[U32, SLAB](0)
    var cb = SIMD[U32, SLAB](0)
    var nw = SIMD[U32, 4 * SLAB](0)
    var ns = SIMD[U32, SLAB](0)
    var nb = SIMD[U32, SLAB](0)
    slab(cw, cs, cb, words, scales, biases, item, gs, items, blocks, gps, kg, n, lc, h)
    var acc = SIMD[F32, 8 * MT](0)  # tile t: lanes 8t .. 8t + 7
    # the slab's rows and group sums, loaded a step ahead and before that step's weights: the loads return in order,
    # so staging never waits on the weights' memory latency
    comptime XN = MT * 16 * SLAB * 8 // THREADS  # uint4 a thread stages
    var xv = SIMD[U32, 4 * XN](0)
    var sv = Float32(0)
    stage[MT](xv, sv, x, xs, ld, m, kg, gs, min(UInt32(SLAB), min(kg, (item // blocks) * gps + gps) - gs), tid)
    while True:
        var slice = item // blocks
        var g1 = min(kg, slice * gps + gps)
        var groups = min(UInt32(SLAB), g1 - gs)
        barrier()  # the previous slab is consumed
        comptime for q in range(XN):
            var i = tid + UInt32(q * THREADS)
            (xsh + Int((i // (SLAB * 8) * XROW + i % (SLAB * 8)) * 4)).store[alignment=16](
                xv.slice[4, offset=4 * q]()
            )
        if tid < UInt32(MT * 16 * SLAB):
            xssh[Int(tid)] = sv
        barrier()
        # the step after (item, gs) in this block's schedule: the next slab, else the next item's first
        var nit: UInt32
        var ngs: UInt32
        if gs + SLAB < g1:
            nit = item
            ngs = gs + SLAB
        else:
            nit = item + nblocks
            ngs = (nit // blocks) * gps
        if nit < items:
            stage[MT](xv, sv, x, xs, ld, m, kg, ngs, min(UInt32(SLAB), min(kg, (nit // blocks) * gps + gps) - ngs), tid)
        slab(nw, ns, nb, words, scales, biases, nit, ngs, items, blocks, gps, kg, n, lc, h)  # a slab ahead
        var p = SIMD[F32, 8 * MT * SLAB](0)  # tile t, group j: lanes 8(t SLAB + j) ..
        comptime for s in range(4):
            comptime for j in range(SLAB):  # independent chains: groups' WMMAs interleave
                var b = pairs(cw[4 * j + s])
                comptime for t in range(MT):
                    var a = (xsh + Int(((UInt32(16 * t) + c) * XROW + UInt32(j * 8 + s) + 4 * h) * 4)).load[
                        width=4, alignment=16
                    ]()
                    comptime at = 8 * (t * SLAB + j)
                    var d = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16", SIMD[F32, 8]](
                        bitcast[DType.int16, 8](a), bitcast[DType.int16, 8](b), p.slice[8, offset=at]()
                    )
                    p = p.insert[offset=at](d)
                comptime if MT > 1:
                    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))
            # keep a K step's A reads with its WMMAs (two tiles: a group's): hoisted all at once they hold 16 * MT uint4
            # (64 VGPRs a tile), which costs the one-tile kernel its second block per CU and spills the two-tile one
            llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))
        comptime for j in range(SLAB):  # then the groups in order
            if UInt32(j) < groups:
                var sc = bf16_value(cs[j])
                var bias = Float32(-128.0).fma(sc, bf16_value(cb[j]))
                comptime for t in range(MT):
                    var xv = SIMD[F32, 8](0)
                    comptime for i in range(8):
                        xv[i] = xssh[Int(UInt32((16 * t + i) * SLAB + j) + 8 * SLAB * h)]
                    var pv = p.slice[8, offset=8 * (t * SLAB + j)]()
                    var av = acc.slice[8, offset=8 * t]()
                    acc = acc.insert[offset=8 * t](xv.fma(SIMD[F32, 8](bias), pv.fma(SIMD[F32, 8](sc), av)))
        cw = nw
        cs = ns
        cb = nb
        if nit == item:
            gs = ngs
            continue
        # the item is done: its output, or its slice's sums and perhaps the column's
        var col = (item % blocks) * COLS + lc
        var live = col < n
        if slices == 1:
            if live:
                comptime for t in range(MT):
                    comptime for i in range(8):
                        var r = UInt32(16 * t + i) + 8 * h
                        if r < m:
                            var at = Int(r * n + col)
                            if f32 != 0:
                                dst.bitcast[Float32]()[at] = acc[8 * t + i]
                            else:
                                dst[at] = bf16_round(acc[8 * t + i])
        else:
            if live:
                comptime for t in range(MT):
                    comptime for i in range(8):
                        var r = UInt32(16 * t + i) + 8 * h
                        if r < m:
                            part[Int((slice * m + r) * n + col)] = acc[8 * t + i]
            # release: this item's stores only (a full fence would also wait out the next item's prefetched loads)
            fence[ordering=Ordering.RELEASE, scope="agent"]()
            barrier()
            if tid == 0:
                var before = Atomic[Int32, scope="agent"].fetch_add[ordering=Ordering.RELAXED](
                    counts + Int(item % blocks), Int32(1)
                )
                last[0] = Int32(1) if UInt32(before) == slices - 1 else Int32(0)
            barrier()
            if last[0] != 0:
                fence[ordering=Ordering.ACQUIRE, scope="agent"]()
                if live:
                    comptime for t in range(MT):
                        comptime for i in range(8):
                            var r = UInt32(16 * t + i) + 8 * h
                            if r < m:
                                var at = Int(r * n + col)
                                var total = Atomic[Float32, scope="agent"].load[ordering=Ordering.RELAXED](
                                    part + at
                                )
                                for u in range(1, Int(slices)):
                                    total = total + Atomic[Float32, scope="agent"].load[
                                        ordering=Ordering.RELAXED
                                    ](part + Int((UInt32(u) * m + r) * n + col))
                                if f32 != 0:
                                    dst.bitcast[Float32]()[at] = total
                                else:
                                    dst[at] = bf16_round(total)
                if tid == 0:
                    counts[Int(item % blocks)] = 0
        item = nit
        if item >= items:
            break
        gs = ngs
        acc = SIMD[F32, 8 * MT](0)


# One or two 16-row tiles (a pass of up to 16 rows, or up to 32).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](THREADS))
def wmma_mt1(
    x: U32Ptr,
    ldx2: Int32,
    m: Int32,
    xs: FPtr,
    kg: Int32,
    gps: Int32,
    slices: Int32,
    words: U32Ptr,
    scales: U16Ptr,
    biases: U16Ptr,
    n: Int32,
    part: FPtr,
    dst: U16Ptr,
    f32: Int32,
    counts: IPtr,
    nblocks: Int32,
):
    wmma_body[1](x, ldx2, m, xs, kg, gps, slices, words, scales, biases, n, part, dst, f32, counts, nblocks)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](THREADS))
def wmma_mt2(
    x: U32Ptr,
    ldx2: Int32,
    m: Int32,
    xs: FPtr,
    kg: Int32,
    gps: Int32,
    slices: Int32,
    words: U32Ptr,
    scales: U16Ptr,
    biases: U16Ptr,
    n: Int32,
    part: FPtr,
    dst: U16Ptr,
    f32: Int32,
    counts: IPtr,
    nblocks: Int32,
):
    wmma_body[2](x, ldx2, m, xs, kg, gps, slices, words, scales, biases, n, part, dst, f32, counts, nblocks)


# K slices added in slice order: part (slices, total) fp32 -> out fp32 (f32) or bf16. 256-thread blocks.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def reduce_kernel(part: FPtr, slices: Int32, total: Int64, out32: FPtr, out16: U16Ptr, f32: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i >= Int(total):
        return
    var acc = part[i]
    for s in range(1, Int(slices)):
        acc = acc + part[s * Int(total) + i]
    if f32 != 0:
        out32[i] = acc
    else:
        out16[i] = bf16_round(acc)


@export
def tf_qmm_instantiate(a: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var u = U32Ptr(unsafe_from_address=a)
        var s = U16Ptr(unsafe_from_address=a)
        var f = FPtr(unsafe_from_address=a)
        var i = IPtr(unsafe_from_address=a)
        var z = Int32(0)
        ctx.enqueue_function[wmma_mt1](u, z, z, f, z, z, z, u, s, s, z, f, s, z, i, z, grid_dim=1, block_dim=THREADS)
        ctx.enqueue_function[wmma_mt2](u, z, z, f, z, z, z, u, s, s, z, f, s, z, i, z, grid_dim=1, block_dim=THREADS)
        ctx.enqueue_function[reduce_kernel](f, z, Int64(0), f, s, z, grid_dim=1, block_dim=256)
        return 0
    except:
        return 1
