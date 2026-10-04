# Check 2: vector add on gfx1201 through DeviceContext.
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv

comptime N = 1 << 20
comptime BLOCK = 256


def vecadd(a: UnsafePointer[Float32, MutAnyOrigin],
           b: UnsafePointer[Float32, MutAnyOrigin],
           c: UnsafePointer[Float32, MutAnyOrigin],
           n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        c[i] = a[i] + b[i]


def main() raises:
    with DeviceContext() as ctx:
        print("device:", ctx.name(), "api:", ctx.api())
        var ha = ctx.enqueue_create_host_buffer[DType.float32](N)
        var hb = ctx.enqueue_create_host_buffer[DType.float32](N)
        var hc = ctx.enqueue_create_host_buffer[DType.float32](N)
        for i in range(N):
            ha[i] = Float32(i)
            hb[i] = Float32(2 * i)
        var da = ctx.enqueue_create_buffer[DType.float32](N)
        var db = ctx.enqueue_create_buffer[DType.float32](N)
        var dc = ctx.enqueue_create_buffer[DType.float32](N)
        ctx.enqueue_copy(da, ha)
        ctx.enqueue_copy(db, hb)
        ctx.enqueue_function[vecadd](
            da, db, dc, Int32(N), grid_dim=ceildiv(N, BLOCK), block_dim=BLOCK
        )
        ctx.enqueue_copy(hc, dc)
        ctx.synchronize()
        var bad = 0
        for i in range(N):
            if hc[i] != Float32(3 * i):
                bad += 1
        print("mismatches:", bad)
        print("PASS" if bad == 0 else "FAIL")
