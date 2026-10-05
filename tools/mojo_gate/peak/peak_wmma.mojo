# The R9700's sustained WMMA rate: each wave runs 16 independent chains of v_wmma_f32_16x16x16_bf16 (or the fp8
# form) on register operands, K / 16 steps each; dst takes the sums so nothing folds away. Same signature as the
# prompt GEMM kernels, so prefill_rocm_mojo.cu launches it (grid m x n blocks of 256 threads).
from std.sys import llvm_intrinsic
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu import thread_idx, block_idx, MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.host import DeviceContext

comptime F32 = DType.float32
comptime U32 = DType.uint32
comptime U16Ptr = UnsafePointer[UInt16, MutAnyOrigin]


@always_inline
def body[FP8: Bool](dst: U16Ptr, k32: Int32):
    var t = UInt32(thread_idx.x)
    var a = SIMD[U32, 4](t, t + 1, t + 2, t + 3) & 0x3F003F00
    var acc = SIMD[F32, 128](0)
    for _ in range(Int(k32) // 16):
        comptime for i in range(16):
            var b = a + UInt32(i)
            comptime if FP8:
                var d = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32", SIMD[F32, 8]](
                    bitcast[DType.int32, 2](a.slice[2]()), bitcast[DType.int32, 2](b.slice[2]()),
                    acc.slice[8, offset=8 * i]()
                )
                acc = acc.insert[offset=8 * i](d)
            else:
                var d = llvm_intrinsic["llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16", SIMD[F32, 8]](
                    bitcast[DType.int16, 8](a), bitcast[DType.int16, 8](b), acc.slice[8, offset=8 * i]()
                )
                acc = acc.insert[offset=8 * i](d)
    var s = Float32(0)
    comptime for i in range(128):
        s += acc[i]
    if s == Float32(1.2345):
        dst[Int(t)] = 1


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def peak_bf16(x: U16Ptr, w: U16Ptr, dst: U16Ptr, m: Int32, n: Int32, k: Int32, f32: Int32):
    body[False](dst, k)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def peak_fp8(x: U16Ptr, w: U16Ptr, dst: U16Ptr, m: Int32, n: Int32, k: Int32, f32: Int32):
    body[True](dst, k)


@export
def tf_peak_instantiate(a: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var u = U16Ptr(unsafe_from_address=a)
        var z = Int32(0)
        ctx.enqueue_function[peak_bf16](u, u, u, z, z, z, z, grid_dim=1, block_dim=256)
        ctx.enqueue_function[peak_fp8](u, u, u, z, z, z, z, grid_dim=1, block_dim=256)
        return 0
    except:
        return 1
