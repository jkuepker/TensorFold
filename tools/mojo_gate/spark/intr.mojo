# Phase 4 check 4: sm_121 (GB10) tensor-core and memory intrinsics, one warp per test, exported C ABI for ctypes.
#   (a) mma.sync m16n8k16 bf16   (b) ldmatrix   (c) cp.async   (d) padded shared tile   ((e) fp8 / fp4 mma: fp8.mojo)
from std.sys import llvm_intrinsic, inlined_assembly
from std.memory import stack_allocation, bitcast
from max.gpu import thread_idx, barrier
from max.gpu.memory import AddressSpace, async_copy, async_copy_commit_group, async_copy_wait_all
from max.gpu.compute.mma import mma, ld_matrix
from max.gpu.host import DeviceContext

comptime BF = DType.bfloat16
comptime F32 = DType.float32
comptime Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime U8Ptr = UnsafePointer[UInt8, MutAnyOrigin]


# (a) C[16x8 f32] = A[16x16 bf16, row] @ B[16x8 bf16, stored k-major as B[k][n]]; fragments straight from global
def mma_direct(a: Ptr, b: Ptr, c: FPtr):
    var l = Int(thread_idx.x)
    var g = l // 4
    var t = (l % 4) * 2
    var af = SIMD[BF, 8]()
    var bf = SIMD[BF, 4]()
    comptime for i in range(2):
        af[i] = a[g * 16 + t + i]
        af[2 + i] = a[(g + 8) * 16 + t + i]
        af[4 + i] = a[g * 16 + t + 8 + i]
        af[6 + i] = a[(g + 8) * 16 + t + 8 + i]
        bf[i] = b[(t + i) * 8 + g]
        bf[2 + i] = b[(t + 8 + i) * 8 + g]
    var d = SIMD[F32, 4](0)
    mma(d, af, bf, SIMD[F32, 4](0))
    c[g * 8 + t] = d[0]
    c[g * 8 + t + 1] = d[1]
    c[(g + 8) * 8 + t] = d[2]
    c[(g + 8) * 8 + t + 1] = d[3]


# (b)+(c)+(d): cp.async 16 B global->shared into row-padded tiles, ldmatrix (x4 for A, x2.trans for B), mma
comptime A_STRIDE = 24  # 16 + 8 bf16: one 16 B pad per 32 B row
comptime B_STRIDE = 16  # 8 + 8 bf16
def mma_ldm_cpasync(a: Ptr, b: Ptr, c: FPtr):
    var l = Int(thread_idx.x)
    var sa = stack_allocation[16 * A_STRIDE, BFloat16, address_space=AddressSpace.SHARED, alignment=16]()
    var sb = stack_allocation[16 * B_STRIDE, BFloat16, address_space=AddressSpace.SHARED, alignment=16]()
    # A: 16 rows x 2 chunks of 8 bf16 = 32 chunks, one per lane; B: 16 rows x 1 chunk = 16 chunks
    async_copy[16]((a + (l // 2) * 16 + (l % 2) * 8).address_space_cast[AddressSpace.GLOBAL](), sa + (l // 2) * A_STRIDE + (l % 2) * 8)
    if l < 16:
        async_copy[16]((b + l * 8).address_space_cast[AddressSpace.GLOBAL](), sb + l * B_STRIDE)
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    var af = ld_matrix[8](sa + ((l % 8) + ((l // 8) % 2) * 8) * A_STRIDE + (l // 16) * 8)
    var bf = ld_matrix[4, transpose=True](sb + (l % 16) * B_STRIDE)
    var d = SIMD[F32, 4](0)
    mma(d, af, bf, SIMD[F32, 4](0))
    var g = l // 4
    var t = (l % 4) * 2
    c[g * 8 + t] = d[0]
    c[g * 8 + t + 1] = d[1]
    c[(g + 8) * 8 + t] = d[2]
    c[(g + 8) * 8 + t + 1] = d[3]


def _addr[T: AnyType](a: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=a)


@export
def gate_mma(a: Int, b: Int, c: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[mma_direct](_addr[BFloat16](a), _addr[BFloat16](b), _addr[Float32](c), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1


@export
def gate_mma_ldm(a: Int, b: Int, c: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[mma_ldm_cpasync](_addr[BFloat16](a), _addr[BFloat16](b), _addr[Float32](c), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1
