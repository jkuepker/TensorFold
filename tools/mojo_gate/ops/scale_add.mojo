# Check 3 path A: bf16 o = 2*x + y as a max.experimental.torch custom op.
from max.gpu import global_idx
from std.math import ceildiv
from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor, register


def scale_add_kernel(
    o: UnsafePointer[BFloat16, MutAnyOrigin],
    x: UnsafePointer[BFloat16, MutAnyOrigin],
    y: UnsafePointer[BFloat16, MutAnyOrigin],
    n: Int32,
):
    var i = Int(global_idx.x)
    if i < Int(n):
        o[i] = x[i] * 2 + y[i]


@register("scale_add")
struct ScaleAdd:
    @staticmethod
    def execute[
        target: StaticString
    ](
        o: OutputTensor[dtype = DType.bfloat16, rank=1, static_spec=_],
        x: InputTensor[dtype = DType.bfloat16, rank=1, static_spec=_],
        y: InputTensor[dtype = DType.bfloat16, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        var n = o.dim_size(0)
        ctx.enqueue_function[scale_add_kernel](
            o.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(n),
            grid_dim=ceildiv(n, 256), block_dim=256,
        )
