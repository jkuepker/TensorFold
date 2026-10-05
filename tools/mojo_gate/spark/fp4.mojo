# Phase 4 check 4e: block-scaled FP4 (e2m1 x e2m1, ue4m3 scales, 16-element blocks) mma.sync m16n8k64 on sm_121.
# Mojo's mma() has no wrapper for it, so it is inline PTX; the asm stores its own result (one asm output is easier
# than a four-register pack). Layout-agnostic test: every operand nibble is 1.0, so D = sum of the four scale-blocks.
from std.sys import inlined_assembly
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace

comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime U32Ptr = UnsafePointer[UInt32, MutAnyOrigin]


def mma_fp4(a_reg: UInt32, b_reg: UInt32, sfa: UInt32, sfb: UInt32, dst: FPtr):
    var l = Int(thread_idx.x)
    inlined_assembly[
        "{ .reg .f32 d<4>, c<4>; mov.f32 c0, 0f00000000; mov.f32 c1, 0f00000000; mov.f32 c2, 0f00000000; mov.f32 c3, 0f00000000; "
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{d0,d1,d2,d3}, {$1,$1,$1,$1}, {$2,$2}, {c0,c1,c2,c3}, $3, {0, 0}, $4, {0, 0}; "
        "st.global.v4.f32 [$0], {d0,d1,d2,d3}; }",
        NoneType, constraints="l,r,r,r,r", has_side_effect=True,
    ]((dst + l * 4).address_space_cast[AddressSpace.GLOBAL](), a_reg, b_reg, sfa, sfb)


def _addr[T: AnyType](a: Int) -> UnsafePointer[T, MutAnyOrigin]:
    return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=a)


@export
def gate_mma_fp4(sfa: Int, sfb: Int, dst: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function[mma_fp4](UInt32(0x22222222), UInt32(0x22222222), UInt32(sfa), UInt32(sfb), _addr[Float32](dst), grid_dim=1, block_dim=32)
        ctx.synchronize()
        return 0
    except:
        return 1
