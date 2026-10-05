# The ROCm prompt GEMM (qmm_groups.py's Triton ``_gemm``: out = x @ w.T, w the bf16 weight ``dequantize`` rounded once)
# in Mojo for gfx12 (wave32), launched from prefill_rocm_mojo.cu with hipModuleLaunchKernel
# (TF_ROCM_PREFILL_GEMM=mojo). A row gets Triton's bits: every output is one fp32 chain over K of
# v_wmma_f32_16x16x16_bf16, zero start, K steps of 16 in order, the weight as the A operand and the rows as B (as
# Triton's transposed WMMA layout has it), then bf16 nearest even (NaN as 0x7FFF) or fp32. The tile shape never
# changes a bit, only the speed.
#
# Kernels are top-level defs (one per instantiation: cuda/mojo.py names a code object after its def) and must be
# instantiated by a host function, which is what tf_prefill_instantiate is for; it is never called.
from std.sys import llvm_intrinsic
from std.memory import bitcast, stack_allocation
from std.utils import StaticTuple
from max.gpu import thread_idx, block_idx, barrier, MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext

comptime F32 = DType.float32
comptime U32 = DType.uint32
comptime U16Ptr = UnsafePointer[UInt16, MutAnyOrigin]
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime SPtr = UnsafePointer[UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]

@always_inline
def bf16_round(f: Float32) -> UInt32:
    var u = bitcast[U32, 1](f)
    return UInt32(0x7FFF) if f != f else (u + 0x7FFF + ((u >> 16) & 1)) >> 16


@always_inline
def wmma(a: SIMD[U32, 4], b: SIMD[U32, 4], c: SIMD[F32, 8]) -> SIMD[F32, 8]:
    return llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16", SIMD[F32, 8]](
        bitcast[DType.int16, 8](a), bitcast[DType.int16, 8](b), c
    )


# 8 bf16 at element ``off`` of a wave-uniform base (one global_load_b128, the base in SGPRs).
@always_inline
def ld4(p: U16Ptr, off: UInt32) -> SIMD[U32, 4]:
    return (p.bitcast[UInt8]() + Int(off * 2)).bitcast[UInt32]().load[width=4, alignment=16]()


# R rows (from row r0 of ``p``, ``lim`` of them real) over KS of K into registers: 16-byte pieces, KS / 8 a row,
# the pieces a thread takes T apart (T / (KS / 8) rows apart). Rows past ``lim`` read the last real one: an output
# depends on its own row and column only, and theirs are never stored.
@always_inline
def fetch[R: Int, KS: Int, T: Int](
    mut v: SIMD[U32, 4 * (R * KS // 8 // T)], p: U16Ptr, r0: UInt32, lim: UInt32, k: UInt32, k0: UInt32, tid: UInt32
):
    comptime PR = KS // 8
    var r = r0 + tid // UInt32(PR)
    var col = k0 + (tid % UInt32(PR)) * 8
    comptime for j in range(R * KS // 8 // T):
        v = v.insert[offset=4 * j](ld4(p, min(r + UInt32(j * (T // PR)), lim - 1) * k + col))


# The registers ``fetch`` filled into a staged tile (rows of KS + 8 bf16: 8 of padding, so the 16 rows a fragment read
# takes miss each other's banks); one address a thread, the pieces at immediate offsets.
@always_inline
def put[R: Int, KS: Int, T: Int](t: SPtr, v: SIMD[U32, 4 * (R * KS // 8 // T)], tid: UInt32):
    comptime PR = KS // 8
    var at = t + Int(((tid // UInt32(PR)) * UInt32(KS + 8) + (tid % UInt32(PR)) * 8) // 2)
    comptime for j in range(R * KS // 8 // T):
        (at + j * (T // PR) * (KS + 8) // 2).store[alignment=16](v.slice[4, offset=4 * j]())


# A K step's fragments for a wave: FN of the weight (A), then FM of the inputs (B), each at an immediate offset from
# the lane's first fragment row (``xa``: input row wm + c, ``wa``: output wn + c, both at K 8h).
@always_inline
def frags[KS: Int, FM: Int, FN: Int, S: Int](xa: SPtr, wa: SPtr) -> SIMD[U32, 4 * (FM + FN)]:
    comptime ROW = KS + 8
    var f = SIMD[U32, 4 * (FM + FN)](0)
    comptime for a in range(FN):
        f = f.insert[offset=4 * a]((wa + (16 * a * ROW + 16 * S) // 2).load[width=4, alignment=16]())
    comptime for b in range(FM):
        f = f.insert[offset=4 * (FN + b)]((xa + (16 * b * ROW + 16 * S) // 2).load[width=4, alignment=16]())
    return f


# A wave's WMMAs over one staged step: KS / 16 K steps in order, each output's chain one WMMA a step; the next K
# step's fragments are read before this one's WMMAs (sched barriers keep them there), so the WMMAs never wait on LDS.
@always_inline
def steps[KS: Int, FM: Int, FN: Int](mut acc: SIMD[F32, 8 * FM * FN], xa: SPtr, wa: SPtr):
    var cur = frags[KS, FM, FN, 0](xa, wa)
    comptime for s in range(KS // 16):
        var nxt = cur
        comptime if s + 1 < KS // 16:
            nxt = frags[KS, FM, FN, s + 1](xa, wa)
        llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))
        comptime for a in range(FN):
            comptime for b in range(FM):
                comptime at = 8 * (a * FM + b)
                acc = acc.insert[offset=at](
                    wmma(cur.slice[4, offset=4 * a](), cur.slice[4, offset=4 * (FN + b)](), acc.slice[8, offset=at]())
                )
        llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))
        cur = nxt


# A block: BM rows by BN outputs, waves of TM rows by TN outputs (WAVES = BM/TM * BN/TN), K staged KS at a time in
# one LDS buffer; the next step's global loads are issued before this step's WMMAs and staged after them. Lane l of
# a wave: A (the weight) row and B (the input) column l % 16, half h = l / 16 holding K 8h .. 8h + 7 of a 16-step;
# D holds input row l % 16 and outputs 8h .. 8h + 7, eight contiguous outputs a lane. (Two LDS buffers with one
# barrier a step need KS = 32 to fit 64 KB: measured slower.)
@always_inline
def gemm_body[BM: Int, BN: Int, TM: Int, TN: Int, KS: Int](
    x: U16Ptr, w: U16Ptr, dst: U16Ptr, m32: Int32, n32: Int32, k32: Int32, f32: Int32
):
    comptime WM = BM // TM  # waves along the rows
    comptime T = 32 * WM * (BN // TN)
    comptime XP = BM * KS // 8 // T
    comptime WP = BN * KS // 8 // T
    comptime FM = TM // 16  # B fragments (16 input rows each)
    comptime FN = TN // 16  # A fragments (16 outputs each)
    comptime TILE = (BM + BN) * (KS + 8) // 2  # uint32 a staged step: x rows, then w rows
    var lds = stack_allocation[TILE, UInt32, alignment=16, address_space=AddressSpace.SHARED]()
    var m = UInt32(m32)
    var n = UInt32(n32)
    var k = UInt32(k32)
    var tid = UInt32(thread_idx.x)
    var lane = tid & 31
    var wave = tid >> 5
    var c = lane & 15
    var h = lane >> 4
    var m0 = UInt32(block_idx.x) * UInt32(BM)
    var n0 = UInt32(block_idx.y) * UInt32(BN)
    var wm = (wave % UInt32(WM)) * UInt32(TM)  # the wave's first row and output within the block
    var wn = (wave // UInt32(WM)) * UInt32(TN)
    var acc = SIMD[F32, 8 * FM * FN](0)  # fragment (a, b) at 8 (a FM + b)
    var xr = SIMD[U32, 4 * XP](0)
    var wr = SIMD[U32, 4 * WP](0)
    var nsteps = k // UInt32(KS)
    var lx = Int(((wm + c) * UInt32(KS + 8) + 8 * h) // 2)  # the lane's fragment rows in a staged step
    var lw = BM * (KS + 8) // 2 + Int(((wn + c) * UInt32(KS + 8) + 8 * h) // 2)
    fetch[BM, KS, T](xr, x, m0, m, k, 0, tid)
    fetch[BN, KS, T](wr, w, n0, n, k, 0, tid)
    for kt in range(Int(nsteps)):
        barrier()  # the previous step's fragments are read
        put[BM, KS, T](lds, xr, tid)
        put[BN, KS, T](lds + BM * (KS + 8) // 2, wr, tid)
        barrier()
        if UInt32(kt) + 1 < nsteps:
            fetch[BM, KS, T](xr, x, m0, m, k, UInt32(kt + 1) * UInt32(KS), tid)
            fetch[BN, KS, T](wr, w, n0, n, k, UInt32(kt + 1) * UInt32(KS), tid)
        steps[KS, FM, FN](acc, lds + lx, lds + lw)
    # outputs: input row m0 + wm + 16 b + c, outputs n0 + wn + 16 a + 8 h .. + 7
    comptime for a in range(FN):
        comptime for b in range(FM):
            comptime at = 8 * (a * FM + b)
            var r = m0 + wm + UInt32(16 * b) + c
            var col = n0 + wn + UInt32(16 * a) + 8 * h
            if r < m:
                var o = r * n + col
                if f32 != 0:
                    var d32 = dst.bitcast[Float32]()
                    if col + 8 <= n and (n & 3) == 0:
                        (d32 + Int(o)).store[alignment=16](acc.slice[4, offset=at]())
                        (d32 + Int(o + 4)).store[alignment=16](acc.slice[4, offset=at + 4]())
                    else:
                        comptime for i in range(8):
                            if col + UInt32(i) < n:
                                d32[Int(o) + i] = acc[at + i]
                else:
                    if col + 8 <= n and (n & 7) == 0:
                        var p = SIMD[U32, 4](0)
                        comptime for i in range(4):
                            p[i] = bf16_round(acc[at + 2 * i]) | (bf16_round(acc[at + 2 * i + 1]) << 16)
                        (dst + Int(o)).bitcast[UInt32]().store[alignment=16](p)
                    else:
                        comptime for i in range(8):
                            if col + UInt32(i) < n:
                                dst[Int(o) + i] = UInt16(bf16_round(acc[at + i]))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
def gemm_128x128(x: U16Ptr, w: U16Ptr, dst: U16Ptr, m: Int32, n: Int32, k: Int32, f32: Int32):
    gemm_body[128, 128, 64, 64, 64](x, w, dst, m, n, k, f32)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def gemm_128x256(x: U16Ptr, w: U16Ptr, dst: U16Ptr, m: Int32, n: Int32, k: Int32, f32: Int32):
    gemm_body[128, 256, 64, 64, 64](x, w, dst, m, n, k, f32)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def gemm_256x128(x: U16Ptr, w: U16Ptr, dst: U16Ptr, m: Int32, n: Int32, k: Int32, f32: Int32):
    gemm_body[256, 128, 64, 64, 64](x, w, dst, m, n, k, f32)


@export
def tf_prefill_instantiate(a: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var u = U16Ptr(unsafe_from_address=a)
        var z = Int32(0)
        ctx.enqueue_function[gemm_128x128](u, u, u, z, z, z, z, grid_dim=1, block_dim=128)
        ctx.enqueue_function[gemm_128x256](u, u, u, z, z, z, z, grid_dim=1, block_dim=256)
        ctx.enqueue_function[gemm_256x128](u, u, u, z, z, z, z, grid_dim=1, block_dim=256)
        return 0
    except:
        return 1
