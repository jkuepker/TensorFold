# Check 5 (GB10): stream-read bandwidth. 16-byte loads, grid-stride, sum-reduce (per-thread partials).
from max.gpu import global_idx, grid_dim, block_dim
from max.gpu.host import DeviceContext
from std.memory import bitcast

comptime U32x4 = SIMD[DType.uint32, 4]
comptime U32P = UnsafePointer[UInt32, MutAnyOrigin]


def sum16_kernel(src: U32P, partial: U32P, nvec: Int64):
    var gid = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var acc = U32x4(0)
    var i = gid
    var n = Int(nvec)
    while i < n:
        acc += (src + i * 4).load[width=4, alignment=16](0)
        i += stride
    partial[gid] = acc[0] + acc[1] + acc[2] + acc[3]


@export
def gate_bw(src: Int, partial: Int, nbytes: Int, stream: Int, grid: Int, reps: Int) abi("C") -> Int:
    """Enqueue `reps` sum16 launches on the caller's CUstream (no sync)."""
    try:
        var ctx = DeviceContext()
        var f = ctx.compile_function[sum16_kernel]()
        var s = ctx.create_external_stream(
            Optional(UnsafePointer[NoneType, MutAnyOrigin](unsafe_from_address=stream)))
        var ps = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=src)
        var pp = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=partial)
        for _ in range(reps):
            s.enqueue_function(f, ps, pp, Int64(nbytes // 16), grid_dim=grid, block_dim=256)
        return 0
    except:
        return 1
