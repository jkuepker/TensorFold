# Phase 4 check 4e: fp8 e4m3 mma.sync m16n8k32 and block-scaled fp4 mma.sync on sm_121, via the mma API and inline PTX.
from std.sys import inlined_assembly
from std.memory import bitcast
from max.gpu import thread_idx
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext

comptime F32 = DType.float32
comptime E4 = DType.float8_e4m3fn
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]


# C[16x8 f32] = A[16x32 e4m3, row] @ B[32x8 e4m3, stored k-major]. A frag: 16 e4m3 (a0..a3 regs: (g,k=t*4..), (g+8,..),
# (g,k+16..), (g+8,k+16..)); B frag: 8 e4m3 ((k=t*4.., n=g), (k+16.., n=g)), t = lane % 4.
def mma_fp8_api(a: U8Ptr, b: U8Ptr, c: FPtr):
    var l = Int(thread_idx.x)
    var g = l // 4
    var t = (l % 4) * 4
    var af = SIMD[E4, 16]()
    var bf = SIMD[E4, 8]()
    comptime for i in range(4):
        af[i] = bitcast[E4, 1](a[g * 32 + t + i])
        af[4 + i] = bitcast[E4, 1](a[(g + 8) * 32 + t + i])
        af[8 + i] = bitcast[E4, 1](a[g * 32 + t + 16 + i])
        af[12 + i] = bitcast[E4, 1](a[(g + 8) * 32 + t + 16 + i])
        bf[i] = bitcast[E4, 1](b[(t + i) * 8 + g])
        bf[4 + i] = bitcast[E4, 1](b[(t + 16 + i) * 8 + g])
    var d = SIMD[F32, 4](0)
    mma(d, af, bf, SIMD[F32, 4](0))
    var tt = (l % 4) * 2
    c[g * 8 + tt] = d[0]
    c[g * 8 + tt + 1] = d[1]
    c[(g + 8) * 8 + tt] = d[2]
    c[(g + 8) * 8 + tt + 1] = d[3]


def _addr[T: AnyType](a: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=a)


@export
def gate_mma_fp8(a: Int, b: Int, c: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[mma_fp8_api](_addr[UInt8](a), _addr[UInt8](b), _addr[Float32](c), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1
