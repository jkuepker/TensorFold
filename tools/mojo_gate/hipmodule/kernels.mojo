# Phase 1a: kernels for the hipModuleLoadData launch spike (gfx1201, wave32).
# gate_all() only exists so `mojo build` instantiates (and so dumps) every kernel; it is never called.
from std.sys import llvm_intrinsic
from std.memory import bitcast
from max.gpu import thread_idx, global_idx
from max.gpu.host import DeviceContext

comptime BF = DType.bfloat16
comptime F32 = DType.float32
comptime Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime I64Ptr = UnsafePointer[Int64, MutAnyOrigin]


# (i) bf16 out = 2*x + y
def scale_add(o: Ptr, x: Ptr, y: Ptr, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        o[i] = x[i] * 2 + y[i]


# (ii) one wave32: C[16x16 f32] = A[16x16 bf16] @ B[16x16 bf16]
def wmma_kernel(a: Ptr, b: Ptr, c: FPtr):
    var l = Int(thread_idx.x)
    var row = l % 16
    var kh = l // 16
    var af = SIMD[BF, 8]()
    var bf = SIMD[BF, 8]()
    comptime for i in range(8):
        af[i] = a[row * 16 + kh * 8 + i]
        bf[i] = b[(kh * 8 + i) * 16 + row]
    var acc = SIMD[F32, 8](0)
    acc = llvm_intrinsic[
        "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16", SIMD[F32, 8]
    ](bitcast[DType.int16, 8](af), bitcast[DType.int16, 8](bf), acc)
    comptime for i in range(8):
        c[(kh * 8 + i) * 16 + row] = acc[i]


# (iii) kernarg-layout probe: mixed Int32/Int64/pointer, writes what it received
def argprobe(dst: I64Ptr, a: Int32, b: Int64, c: Int32, d: Int64):
    dst[0] = Int64(a)
    dst[1] = b
    dst[2] = Int64(c)
    dst[3] = d


@export
def gate_all(a: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var p = Ptr(unsafe_from_address=a)
        var f = FPtr(unsafe_from_address=a)
        var q = I64Ptr(unsafe_from_address=a)
        ctx.enqueue_function[scale_add](p, p, p, Int32(0), grid_dim=1, block_dim=256)
        ctx.enqueue_function[wmma_kernel](p, p, f, grid_dim=1, block_dim=32)
        ctx.enqueue_function[argprobe](q, Int32(1), Int64(2), Int32(3), Int64(4), grid_dim=1, block_dim=1)
        ctx.synchronize()
        return 0
    except:
        return 1
