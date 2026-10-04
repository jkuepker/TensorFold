# Check 4: gfx1201 intrinsics, one 32-lane (wave32) block per test, exported C ABI for ctypes.
from std.sys import llvm_intrinsic
from std.memory import stack_allocation
from std.math import ceildiv
from std.memory import bitcast
from max.gpu import thread_idx, barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext

comptime BF = DType.bfloat16
comptime F32 = DType.float32
comptime Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]


# (a) bf16 WMMA 16x16x16, wave32. C[16x16 f32] = A[16x16 bf16] @ B[16x16 bf16]
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


# (b) global.load.tr.b128 : lane l passes &src[l*8]; record what each lane receives
def loadtr_kernel(src: Ptr, dst: Ptr):
    var l = Int(thread_idx.x)
    var v = llvm_intrinsic[
        "llvm.amdgcn.global.load.tr.b128.v8bf16", SIMD[BF, 8]
    ](( src + l * 8).address_space_cast[AddressSpace.GLOBAL]())
    comptime for i in range(8):
        dst[l * 8 + i] = v[i]


# (c) LDS tile with manual row padding (+1 element of 16 B in the original; here +8 bf16 = 16 B)
comptime ROWS = 16
comptime COLS = 32
comptime PAD = 8  # one 16-byte slot per row, like uint4 xsh[..][SLAB*8+1]
comptime STRIDE = COLS + PAD


def lds_kernel(src: Ptr, dst: Ptr):
    var l = Int(thread_idx.x)
    var tile = stack_allocation[
        ROWS * STRIDE, BFloat16, address_space = AddressSpace.SHARED
    ]()
    # each lane writes 16 bf16 (16 rows x 32 cols = 512 elems / 32 lanes)
    comptime for i in range(16):
        var idx = l * 16 + i
        tile[(idx // COLS) * STRIDE + (idx % COLS)] = src[idx]
    barrier()
    # read transposed: lane l reads column l, all 16 rows
    comptime for r in range(16):
        dst[l * ROWS + r] = tile[r * STRIDE + l]


# (d) non-temporal 16-byte loads
def nt_kernel(src: Ptr, dst: Ptr):
    var l = Int(thread_idx.x)
    var v = (src + l * 8).load[width=8, non_temporal=True]()
    comptime for i in range(8):
        dst[l * 8 + i] = v[i]


# (e) v_dot2_f32_bf16 : out[l] = dot2(a[l], b[l]) + c
def dot2_kernel(a: Ptr, b: Ptr, o: FPtr):
    var l = Int(thread_idx.x)
    var av = SIMD[BF, 2](a[2 * l], a[2 * l + 1])
    var bv = SIMD[BF, 2](b[2 * l], b[2 * l + 1])
    o[l] = llvm_intrinsic[
        "llvm.amdgcn.fdot2.f32.bf16", Float32
    ](av, bv, Float32(0.5), False)


def _addr[T: AnyType](a: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=a)


@export
def gate_wmma(a: Int, b: Int, c: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[wmma_kernel](_addr[BFloat16](a), _addr[BFloat16](b), _addr[Float32](c), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1


@export
def gate_loadtr(s: Int, d: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[loadtr_kernel](_addr[BFloat16](s), _addr[BFloat16](d), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1


@export
def gate_lds(s: Int, d: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[lds_kernel](_addr[BFloat16](s), _addr[BFloat16](d), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1


@export
def gate_nt(s: Int, d: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[nt_kernel](_addr[BFloat16](s), _addr[BFloat16](d), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1


@export
def gate_dot2(a: Int, b: Int, o: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[dot2_kernel](_addr[BFloat16](a), _addr[BFloat16](b), _addr[Float32](o), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1
