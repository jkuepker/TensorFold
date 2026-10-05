# Phase 4 launch spike: kernels for the cuModuleLoadData path (GB10, sm_121).
# gate_all() only exists so `mojo build` instantiates (and so dumps) every kernel; it is never called.
from max.gpu import global_idx
from max.gpu.host import DeviceContext

comptime BF = DType.bfloat16
comptime Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime I64Ptr = UnsafePointer[Int64, MutAnyOrigin]


# bf16 out = 2*x + y
def scale_add(o: Ptr, x: Ptr, y: Ptr, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        o[i] = x[i] * 2 + y[i]


# kernarg-layout probe: mixed Int32/Int64/pointer, writes what it received
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
        var q = I64Ptr(unsafe_from_address=a)
        ctx.enqueue_function[scale_add](p, p, p, Int32(0), grid_dim=1, block_dim=256)
        ctx.enqueue_function[argprobe](q, Int32(1), Int64(2), Int32(3), Int64(4), grid_dim=1, block_dim=1)
        ctx.synchronize()
        return 0
    except:
        return 1
