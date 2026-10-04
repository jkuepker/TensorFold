# Check 3 path B: shared lib with a C-ABI entry taking raw device pointers (as Int)
# plus an optional external hipStream_t, called from Python via ctypes.
from max.gpu import global_idx
from std.math import ceildiv

from max.gpu.host import DeviceContext


def scale_add_kernel(
    o: UnsafePointer[BFloat16, MutAnyOrigin],
    x: UnsafePointer[BFloat16, MutAnyOrigin],
    y: UnsafePointer[BFloat16, MutAnyOrigin],
    n: Int32,
):
    var i = Int(global_idx.x)
    if i < Int(n):
        o[i] = x[i] * 2 + y[i]


@export
def gate_init() abi("C") -> Int:
    """Placeholder handle. A persistent DeviceContext could not be kept across calls:
    Pointer.write needs trivial-del types, OwnedPointer/List have no steal_data, and
    __disable_del is unknown in Mojo 1.1.0, so every call builds its own DeviceContext."""
    return 1


@export
def gate_scale_add(handle: Int, o: Int, x: Int, y: Int, n: Int, stream: Int) abi("C") -> Int:
    """o = 2*x + y over n bf16 elements. stream==0: Mojo's own stream (+sync).
    stream!=0: enqueue on the caller's hipStream_t, no sync."""
    try:
        var ctx = DeviceContext()
        var po = UnsafePointer[BFloat16, MutAnyOrigin](unsafe_from_address=o)
        var px = UnsafePointer[BFloat16, MutAnyOrigin](unsafe_from_address=x)
        var py = UnsafePointer[BFloat16, MutAnyOrigin](unsafe_from_address=y)
        var f = ctx.compile_function[scale_add_kernel]()
        if stream == 0:
            ctx.enqueue_function(f, po, px, py, Int32(n),
                                 grid_dim=ceildiv(n, 256), block_dim=256)
            ctx.synchronize()
        else:
            var s = ctx.create_external_stream(
                Optional(UnsafePointer[NoneType, MutAnyOrigin](unsafe_from_address=stream)))
            s.enqueue_function(f, po, px, py, Int32(n),
                               grid_dim=ceildiv(n, 256), block_dim=256)
        return 0
    except:
        return 1
