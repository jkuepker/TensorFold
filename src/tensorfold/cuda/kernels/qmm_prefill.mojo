# The NVIDIA prompt matmul (qmm_prefill.cu's prefill_kernel, the bf16 prompt default since 0.6.0) in Mojo, for the
# GB10 (sm_121), launched from qmm_prefill_mojo.cu with cuLaunchKernel (TF_CUDA_PREFILL_GEMM=mojo). The arithmetic is
# the CUDA kernel's, instruction for instruction, so the two give the same bits: each weight rounded once to bf16 as
# fma.rn.bf16x2(q, s, b) with q from sub.rn.bf16x2 (128 + q) - 128, one fp32 chain over K of
# mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32 (groups in order, four k16 steps a group), outputs rounded to nearest
# even by cvt.rn. The tile never changes a row's bits (each output's chain is the same at any tile or chunking).
#
# Kernels are top-level defs (one per instantiation: the carve names a PTX module after its def) and must be
# instantiated by a host function, which is what tf_prefill_instantiate is for; it is never called.
from std.sys import inlined_assembly
from std.memory import bitcast, stack_allocation
from std.utils import StaticTuple
from max.gpu import thread_idx, block_idx, barrier, MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.memory import AddressSpace, async_copy, async_copy_commit_group, async_copy_wait_group
from max.gpu.compute.mma import mma, ld_matrix
from max.gpu.host import DeviceContext

comptime BF = DType.bfloat16
comptime F32 = DType.float32
comptime U32 = DType.uint32
comptime BPtr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime U32Ptr = UnsafePointer[UInt32, MutAnyOrigin]
comptime U16Ptr = UnsafePointer[UInt16, MutAnyOrigin]

# The tile prompt_tile picks (qmm.py: 9): 128 x 128 a block on 2 x 2 warps of 64 x 64, two stages, 64-input groups.
comptime GS = 64
comptime BM = 128
comptime BN = 128
comptime WM = 2
comptime WN = 2
comptime STAGES = 2
comptime THREADS = WM * WN * 32
comptime MT = BM // WM // 16  # m16 tiles a warp
comptime NT = BN // WN // 8  # n8 tiles a warp
comptime ROW = GS * 2  # bytes of one input row a group
comptime CHUNKS = ROW // 16
comptime XB = BM * ROW  # stage bytes: inputs,
comptime WB = BN * GS // 2  # weights,
comptime SB = BN * 2  # scales and biases (bf16)
comptime STAGE = XB + WB + 2 * SB
comptime TILE_BYTES = 64 * GS // 2  # one stored 64-column tile's group block


# fma.rn.bf16x2(pair(w >> s), sv, bv): the nibbles at bits [s, s + 4) and [16 + s, 20 + s) as the exact bf16 pair
# (q, q') (sub.rn.bf16x2 of 128 + q and 128), then one bf16 rounding of q s + b. One output per asm.
@always_inline
def weight_pair(w: UInt32, s: UInt32, sv: UInt32, bv: UInt32) -> UInt32:
    var t = ((w >> s) & 0x000F000F) | 0x43004300
    return inlined_assembly[
        "{ .reg .b32 q; sub.rn.bf16x2 q, $1, $4; fma.rn.bf16x2 $0, q, $2, $3; }",
        UInt32,
        constraints="=r,r,r,r,r",
        has_side_effect=False,
    ](t, sv, bv, UInt32(0x43004300))


# cvt.rn.bf16x2.f32 (hi, lo): __floats2bfloat162_rn(lo, hi)'s bits
@always_inline
def bf16x2(lo: Float32, hi: Float32) -> UInt32:
    return inlined_assembly["cvt.rn.bf16x2.f32 $0, $1, $2;", UInt32, constraints="=r,f,f", has_side_effect=False](
        hi, lo
    )


@always_inline
def bf16_rn(v: Float32) -> UInt16:
    return UInt16(bf16x2(v, Float32(0)) & 0xFFFF)


@always_inline
def prefill_body[F32OUT: Bool](
    x: BPtr,
    w: U32Ptr,
    scales: U16Ptr,
    biases: U16Ptr,
    dst: U16Ptr,
    m32: Int32,
    n32: Int32,
    k32: Int32,
    npad32: Int32,
    ldx32: Int32,
    group32: Int32,
):
    var buf = stack_allocation[STAGES * STAGE, UInt8, alignment=128, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var lane = tid & 31
    var warp = tid >> 5
    var wm = warp // WN
    var wn = warp % WN
    var M = Int(m32)
    var N = Int(n32)
    var K = Int(k32)
    var npad = Int(npad32)
    var ldx = Int(ldx32)
    var group = Int(group32)
    var KG = K // GS
    # tile_of: row tiles fastest in bands of ``group`` so blocks in flight share L2
    var b = Int(block_idx.x)
    var rows_t = (M + BM - 1) // BM
    var cols_t = (N + BN - 1) // BN
    var band = group * cols_t
    var first = b // band * group
    var in_band = b % band
    var height = min(group, rows_t - first)
    var m0 = (first + in_band % height) * BM
    var n0 = in_band // height * BN

    # one group's inputs, weights, scales and biases into stage s; rows past M read row M - 1 (computed, never stored)
    @always_inline
    @parameter
    def load(s: Int, g: Int):
        var p = buf + s * STAGE
        comptime for q in range(BM * CHUNKS // THREADS):
            var c = tid + q * THREADS
            var r = c // CHUNKS
            var ch = c % CHUNKS
            var row = min(m0 + r, M - 1)
            async_copy[16](
                (x + row * ldx + g * GS + ch * 8).address_space_cast[AddressSpace.GLOBAL](),
                (p + r * ROW + (ch ^ (r % CHUNKS)) * 16).bitcast[BFloat16](),
            )
        var pw = p + XB
        comptime for q in range((WB // 16 + THREADS - 1) // THREADS):
            var c = tid + q * THREADS
            if c < WB // 16:
                var t = c // (TILE_BYTES // 16)
                var off = c % (TILE_BYTES // 16)
                var tile = (n0 // 64 + t) * KG + g
                if n0 + t * 64 < npad:  # a 256-wide block's last tiles may pass the padded columns
                    async_copy[16](
                        (w + tile * (TILE_BYTES // 4) + off * 4).address_space_cast[AddressSpace.GLOBAL](),
                        (pw + c * 16).bitcast[UInt32](),
                    )
        var ps = pw + WB
        if tid < 2 * (SB // 16):
            var which = tid // (SB // 16)
            var off = tid % (SB // 16)
            var src = (biases if which != 0 else scales) + g * npad + n0 + off * 8
            if n0 + off * 8 < npad:
                async_copy[16](src.address_space_cast[AddressSpace.GLOBAL](), (ps + which * SB + off * 16).bitcast[UInt16]())

    var acc = SIMD[F32, MT * NT * 4](0)
    comptime for s in range(STAGES - 1):
        if s < KG:
            load(s, s)
        async_copy_commit_group()
    for it in range(KG):
        async_copy_wait_group(STAGES - 2)
        barrier()
        var nxt = it + STAGES - 1
        if nxt < KG:
            load(nxt % STAGES, nxt)
        async_copy_commit_group()
        var p = buf + (it % STAGES) * STAGE
        var px = p.bitcast[BFloat16]()
        var pw = (p + XB).bitcast[UInt32]()
        var ps = (p + XB + WB).bitcast[UInt16]()
        var words = SIMD[U32, NT * (GS // 32)](0)
        var sv = SIMD[U32, NT](0)
        var bv = SIMD[U32, NT](0)
        comptime for j in range(NT):
            var wv = (pw + ((wn * NT + j) * 32 + lane) * (GS // 32)).load[width = GS // 32, alignment=8]()
            words = words.insert[offset = j * (GS // 32)](wv)
            var col = wn * (BN // WN) + j * 8 + (lane >> 2)  # this lane's B-fragment column
            sv[j] = UInt32(ps[col]) * 0x10001  # (s, s) and (b, b) as bf16 pairs
            bv[j] = UInt32(ps[BN + col]) * 0x10001
        comptime for kt in range(GS // 16):
            var a = SIMD[BF, 8 * MT](0)
            comptime for i in range(MT):
                var r = wm * (BM // WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8
                var ch = kt * 2 + (lane >> 4)
                a = a.insert[offset = 8 * i](ld_matrix[8](px + (r * ROW + (ch ^ (r % CHUNKS)) * 16) // 2))
            comptime for j in range(NT):
                var wd = words[j * (GS // 32) + kt // 2]
                var b0 = weight_pair(wd, UInt32((kt & 1) * 8), sv[j], bv[j])
                var b1 = weight_pair(wd, UInt32((kt & 1) * 8 + 4), sv[j], bv[j])
                var bf = bitcast[BF, 4](SIMD[U32, 2](b0, b1))
                comptime for i in range(MT):
                    comptime o = (i * NT + j) * 4
                    var d = SIMD[F32, 4](0)
                    mma(d, a.slice[8, offset = 8 * i](), bf, acc.slice[4, offset=o]())
                    acc = acc.insert[offset=o](d)
    async_copy_wait_group(0)
    barrier()
    comptime for i in range(MT):
        comptime for j in range(NT):
            var col = n0 + wn * (BN // WN) + j * 8 + (lane & 3) * 2
            comptime for h in range(2):
                var row = m0 + wm * (BM // WM) + i * 16 + (lane >> 2) + h * 8
                if row < M:
                    comptime o = (i * NT + j) * 4 + 2 * h
                    var v0 = acc[o]
                    var v1 = acc[o + 1]
                    comptime if F32OUT:
                        var o32 = dst.bitcast[Float32]() + row * N + col
                        if col < N:
                            o32[0] = v0
                        if col + 1 < N:
                            o32[1] = v1
                    else:
                        var o16 = dst + row * N + col
                        if col + 1 < N and (N & 1) == 0:
                            o16.bitcast[UInt32]()[0] = bf16x2(v0, v1)
                        else:
                            if col < N:
                                o16[0] = bf16_rn(v0)
                            if col + 1 < N:
                                o16[1] = bf16_rn(v1)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](THREADS))
def prefill_bf16(
    x: BPtr,
    w: U32Ptr,
    scales: U16Ptr,
    biases: U16Ptr,
    dst: U16Ptr,
    m: Int32,
    n: Int32,
    k: Int32,
    npad: Int32,
    ldx: Int32,
    group: Int32,
):
    prefill_body[False](x, w, scales, biases, dst, m, n, k, npad, ldx, group)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](THREADS))
def prefill_f32(
    x: BPtr,
    w: U32Ptr,
    scales: U16Ptr,
    biases: U16Ptr,
    dst: U16Ptr,
    m: Int32,
    n: Int32,
    k: Int32,
    npad: Int32,
    ldx: Int32,
    group: Int32,
):
    prefill_body[True](x, w, scales, biases, dst, m, n, k, npad, ldx, group)


@export
def tf_prefill_instantiate(a: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var bp = BPtr(unsafe_from_address=a)
        var u = U32Ptr(unsafe_from_address=a)
        var s = U16Ptr(unsafe_from_address=a)
        var z = Int32(0)
        ctx.enqueue_function[prefill_bf16](bp, u, s, s, s, z, z, z, z, z, z, grid_dim=1, block_dim=THREADS)
        ctx.enqueue_function[prefill_f32](bp, u, s, s, s, z, z, z, z, z, z, grid_dim=1, block_dim=THREADS)
        return 0
    except:
        return 1
