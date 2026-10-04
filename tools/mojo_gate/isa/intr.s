	.att_syntax
	.file	"intr.mojo"
	.text
	.globl	gate_wmma
	.prefalign	4, .Lfunc_end0, nop
	.type	gate_wmma,@function
gate_wmma:
.Lgate_wmma$local:
	.type	.Lgate_wmma$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4744, %rsp
	.cfi_def_cfa_offset 4800
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 184(%rsp)
	leaq	static_string_3b45d8d0deb8f5b3(%rip), %rax
	leaq	208(%rsp), %rbx
	movl	$4, %esi
	movq	%rdi, 176(%rsp)
	movq	%rdx, 192(%rsp)
	movabsq	$2305843009213693952, %r13
	movq	$0, 40(%rsp)
	movq	%rax, 208(%rsp)
	movq	$3, 216(%rsp)
	movq	%rbx, %rdi
	movq	$0, 224(%rsp)
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB0_2
	movq	216(%rsp), %rcx
	jmp	.LBB0_3
.LBB0_2:
	movl	$1336, %edx
	bextrq	%rdx, %rcx, %rcx
.LBB0_3:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	movq	%rax, %rcx
	orq	%r13, %rcx
	movq	%rcx, 224(%rsp)
	testq	%rax, %rax
	js	.LBB0_5
	movq	208(%rsp), %rbx
.LBB0_5:
	leaq	40(%rsp), %rdi
	movq	%rbx, %rsi
	xorl	%edx, %edx
	callq	AsyncRT_DeviceContext_create@PLT
	testb	$64, 231(%rsp)
	je	.LBB0_8
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_8
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %rbx
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rax
.LBB0_8:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movabsq	$4611686018427387904, %rbp
	movq	%rcx, 632(%rsp)
	movq	$0, 640(%rsp)
	movq	%r13, 648(%rsp)
	testq	%rax, %rax
	je	.LBB0_18
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_69b8ed51c73424d7(%rip), %r9
	leaq	544(%rsp), %rdi
	leaq	640(%rsp), %rdx
	movl	$4010, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$41
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	544(%rsp), %rbx
	movq	560(%rsp), %r12
	movq	568(%rsp), %r14
	movzbl	576(%rsp), %r15d
	movzbl	536(%rsp), %r13d
	testq	%rbp, 648(%rsp)
	je	.LBB0_12
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_12
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_12:
	testb	$1, %r13b
	movabsq	$2305843009213693952, %r13
	je	.LBB0_18
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r12
	je	.LBB0_16
	lock		decq	-8(%rbx)
	jne	.LBB0_16
	addq	$-8, %rbx
	#MEMBARRIER
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_16:
	movl	$1, %ebx
	cmpb	$1, %r15b
	jne	.LBB0_82
.LBB0_17:
	movq	%r14, %rdi
	jmp	.LBB0_81
.LBB0_18:
	movq	40(%rsp), %rbx
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_retain@PLT
	xorl	%edi, %edi
	movq	$0, 40(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	cmpq	$23, %rax
	jg	.LBB0_22
	leaq	208(%rsp), %rdi
	movabsq	$-9223372036854775808, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, 208(%rsp)
	movq	%rax, 224(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movabsq	$4611686018427387904, %r15
	movq	224(%rsp), %rax
	testq	%r13, %rax
	jne	.LBB0_20
.LBB0_27:
	movl	$1336, %r14d
	testq	%rax, %rax
	js	.LBB0_29
	movq	216(%rsp), %rsi
	jmp	.LBB0_30
.LBB0_22:
	movq	%rax, %r14
	addq	$7, %r14
	movabsq	$4611686018427387904, %r15
	movq	%r14, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB0_96
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB0_98
	sarq	$3, %r14
	movq	$1, (%rax)
	addq	$8, %rax
	movq	$0, 4728(%rsp)
	orq	%r15, %r14
	movq	%rax, 208(%rsp)
	movq	$0, 216(%rsp)
	leaq	208(%rsp), %rax
	movq	%r14, 224(%rsp)
	leaq	632(%rsp), %r14
	movq	%rax, 4736(%rsp)
	movq	%r14, %rdi
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4728(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB0_26
	movq	4736(%rsp), %rdi
	leaq	632(%rsp), %r14
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4728(%rsp)
.LBB0_26:
	movq	4736(%rsp), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	224(%rsp), %rax
	testq	%r13, %rax
	je	.LBB0_27
.LBB0_20:
	testq	%rax, %rax
	js	.LBB0_34
.LBB0_21:
	movq	208(%rsp), %rax
	jmp	.LBB0_35
.LBB0_29:
	bextrq	%r14, %rax, %rsi
.LBB0_30:
	incq	%rsi
	leaq	208(%rsp), %rdi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB0_32
	movq	216(%rsp), %rcx
	jmp	.LBB0_33
.LBB0_32:
	bextrq	%r14, %rcx, %rcx
.LBB0_33:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	orq	%r13, %rax
	movq	%rax, 224(%rsp)
	testq	%rax, %rax
	jns	.LBB0_21
.LBB0_34:
	leaq	208(%rsp), %rax
.LBB0_35:
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_4fc62907bcfcc19a(%rip), %rdx
	leaq	static_string_74d279525b544160(%rip), %rcx
	leaq	static_string_bdc25b2a53dd7d0b(%rip), %r8
	leaq	48(%rsp), %rdi
	movl	$4144, %r9d
	movq	%rbx, %rsi
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$-1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_loadFunction@PLT
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	testq	%r15, 224(%rsp)
	je	.LBB0_38
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_38
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r14, %rax
.LBB0_38:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 632(%rsp)
	movq	$0, 640(%rsp)
	movq	%r13, 648(%rsp)
	testq	%rax, %rax
	je	.LBB0_44
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f6c401c515dc2a68(%rip), %r9
	leaq	496(%rsp), %rdi
	leaq	640(%rsp), %rdx
	movl	$170, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$49
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	%r15, %rax
	movq	496(%rsp), %r14
	movq	512(%rsp), %rbp
	movq	520(%rsp), %r12
	movzbl	528(%rsp), %r13d
	movzbl	488(%rsp), %r15d
	testq	%rax, 648(%rsp)
	je	.LBB0_42
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_42
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_42:
	testb	$1, %r15b
	je	.LBB0_46
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movabsq	$4611686018427387904, %r15
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %rbp
	jne	.LBB0_77
	jmp	.LBB0_79
.LBB0_44:
	movq	40(%rsp), %r12
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	je	.LBB0_47
.LBB0_45:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r12, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%r13, %r12
	movzbl	8(%rsp), %r13d
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %rbp
	jne	.LBB0_77
	jmp	.LBB0_79
.LBB0_46:
	movb	%r13b, 8(%rsp)
	movq	%r12, %r13
	movabsq	$4611686018427387904, %r15
	movq	40(%rsp), %r12
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	jne	.LBB0_45
.LBB0_47:
	leaq	600(%rsp), %rcx
	leaq	607(%rsp), %rax
	movq	%r12, 8(%rsp)
	leaq	632(%rsp), %r14
	movq	$1, 632(%rsp)
	movq	$0, 640(%rsp)
	testq	%rcx, %rcx
	movq	%r14, %rdi
	cmovnsq	%rcx, %rax
	setns	%dl
	sarq	$3, %rax
	movq	%rax, %rsi
	negq	%rsi
	shlq	$3, %rsi
	addq	%rcx, %rsi
	movq	%rbx, %rsi
	setne	%cl
	andb	%dl, %cl
	movzbl	%cl, %r12d
	addq	%rax, %r12
	shlq	$3, %r12
	leaq	16(%r12), %r13
	leaq	8(%r12), %r15
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	movq	640(%rsp), %rdx
	cmpq	$5, %rdx
	jne	.LBB0_52
	movq	632(%rsp), %rax
	leaq	static_string_561d9efdd15f277f(%rip), %rcx
	cmpq	%rcx, %rax
	je	.LBB0_54
	movq	%rdx, %rsi
	sarq	$63, %rsi
	andnq	%rdx, %rsi, %rdx
	xorl	%esi, %esi
	.p2align	4
.LBB0_50:
	cmpq	%rsi, %rdx
	je	.LBB0_54
	movzbl	(%rax,%rsi), %edi
	cmpb	(%rsi,%rcx), %dil
	leaq	1(%rsi), %rsi
	je	.LBB0_50
.LBB0_52:
	movq	176(%rsp), %rax
	movq	184(%rsp), %rdi
	movq	192(%rsp), %rsi
	movq	8(%rsp), %r14
	movq	%r12, 208(%rsp)
	movq	%r15, 216(%rsp)
	movq	%r13, 224(%rsp)
	movq	$0, 584(%rsp)
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%rax, (%r12)
	leaq	208(%rsp), %rax
	movq	%rdi, 8(%r12)
	movq	%rsi, 16(%r12)
	movq	%rbx, %rdi
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	movabsq	$2305843009213693952, %rbp
	testq	%rax, %rax
	je	.LBB0_84
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	632(%rsp), %r14
	movq	%rax, 80(%rsp)
	movq	%rdx, 88(%rsp)
	movq	$1, 632(%rsp)
	movq	%rcx, 96(%rsp)
	movq	$0, 640(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	632(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 152(%rsp)
	movq	$0, 168(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 112(%rsp)
	movq	$12, 136(%rsp)
	movq	$1, 24(%rsp)
	movq	$13, 48(%rsp)
	movq	$1, 640(%rsp)
	movq	%rcx, 104(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 120(%rsp)
	movq	%rcx, 128(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 144(%rsp)
	movq	%rcx, 16(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 32(%rsp)
	movq	%rcx, 40(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 56(%rsp)
	movq	%rcx, 632(%rsp)
	movq	%rbp, 648(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	136(%rsp), %r12
	leaq	static_string_cff33790b37cb0fb(%rip), %r13
	leaq	88(%rsp), %r10
	leaq	24(%rsp), %r11
	leaq	160(%rsp), %r15
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	456(%rsp), %rdi
	leaq	112(%rsp), %r9
	movl	$92, %esi
	movl	$42, %edx
	movl	$25, %r8d
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	leaq	64(%rsp), %r10
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r15
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$869
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movzbl	480(%rsp), %r13d
	movq	472(%rsp), %r12
	movq	464(%rsp), %rbp
	movq	448(%rsp), %r14
	movabsq	$4611686018427387904, %r15
	testq	%r15, 648(%rsp)
	jne	.LBB0_56
	jmp	.LBB0_58
.LBB0_54:
	movq	176(%rsp), %rax
	movq	184(%rsp), %rsi
	movq	192(%rsp), %rdi
	movq	%r12, 208(%rsp)
	movq	%r15, 216(%rsp)
	leaq	16(%rsp), %r15
	vxorps	%xmm0, %xmm0, %xmm0
	vmovups	%xmm0, 19(%rsp)
	movb	$0, 16(%rsp)
	movb	$0, 17(%rsp)
	vmovups	%zmm0, 720(%rsp)
	vmovups	%zmm0, 656(%rsp)
	movq	$8, 632(%rsp)
	movq	$8, 640(%rsp)
	movq	%r13, 224(%rsp)
	movq	$8, 648(%rsp)
	movb	$0, 18(%rsp)
	movq	$0, 592(%rsp)
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%rax, (%r12)
	movq	%rsi, 8(%r12)
	leaq	208(%rsp), %rsi
	movq	%rdi, 16(%r12)
	leaq	40(%rsp), %r12
	leaq	200(%rsp), %rax
	movq	%rbx, %rdi
	movq	%rsi, 40(%rsp)
	movq	%r14, 48(%rsp)
	movq	8(%rsp), %r14
	movq	%r15, 56(%rsp)
	movq	$8, 64(%rsp)
	movl	$0, 72(%rsp)
	movq	%r12, 200(%rsp)
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	vzeroupper
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB0_83
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	632(%rsp), %r14
	movq	%rax, 80(%rsp)
	movq	%rdx, 88(%rsp)
	movq	$1, 632(%rsp)
	movq	%rcx, 96(%rsp)
	movq	$0, 640(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	632(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 152(%rsp)
	movq	$0, 168(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 112(%rsp)
	movq	$12, 136(%rsp)
	movq	$1, 24(%rsp)
	movabsq	$2305843009213693952, %rbp
	movq	$13, 48(%rsp)
	movq	$1, 640(%rsp)
	movq	%rcx, 104(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 120(%rsp)
	movq	%rcx, 128(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 144(%rsp)
	movq	%rcx, 16(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 32(%rsp)
	movq	%rcx, 40(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 56(%rsp)
	movq	%rcx, 632(%rsp)
	movq	%rbp, 648(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_cff33790b37cb0fb(%rip), %r13
	movq	%r15, %rbp
	leaq	88(%rsp), %r10
	leaq	160(%rsp), %r11
	leaq	136(%rsp), %r15
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	416(%rsp), %rdi
	leaq	112(%rsp), %r9
	movl	$92, %esi
	movl	$42, %edx
	movl	$25, %r8d
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r15
	.cfi_adjust_cfa_offset 8
	pushq	$869
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movzbl	440(%rsp), %r13d
	movq	432(%rsp), %r12
	movq	424(%rsp), %rbp
	movq	408(%rsp), %r14
	movabsq	$4611686018427387904, %r15
	testq	%r15, 648(%rsp)
	je	.LBB0_58
.LBB0_56:
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_58
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_58:
	testq	%r15, 56(%rsp)
	je	.LBB0_61
	movq	40(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_61
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_61:
	testq	%r15, 32(%rsp)
	je	.LBB0_64
	movq	16(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_64
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_64:
	testq	%r15, 144(%rsp)
	je	.LBB0_67
	movq	128(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_67
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_67:
	testq	%r15, 120(%rsp)
	je	.LBB0_70
	movq	104(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_70
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_70:
	testq	%r15, 168(%rsp)
	je	.LBB0_73
	movq	152(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_73
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_73:
	testq	%r15, 96(%rsp)
	je	.LBB0_76
	movq	80(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_76
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_76:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	8(%rsp), %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %rbp
	je	.LBB0_79
.LBB0_77:
	lock		decq	-8(%r14)
	jne	.LBB0_79
	addq	$-8, %r14
	#MEMBARRIER
	movq	%r14, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_79:
	movl	$1, %ebx
	cmpb	$1, %r13b
	jne	.LBB0_82
	movq	%r12, %rdi
.LBB0_81:
	callq	"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]"@PLT
.LBB0_82:
	movq	%rbx, %rax
	addq	$4744, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_83:
	.cfi_def_cfa_offset 4800
	movabsq	$2305843009213693952, %rbp
.LBB0_84:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_synchronize@PLT
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 632(%rsp)
	movq	$0, 640(%rsp)
	movq	%rbp, 648(%rsp)
	testq	%rax, %rax
	je	.LBB0_94
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f078bd8d2bcbf530(%rip), %r9
	leaq	368(%rsp), %rdi
	leaq	640(%rsp), %rdx
	movl	$93, %ecx
	movl	$24, %r8d
	movq	%rax, %rsi
	pushq	$25
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	368(%rsp), %r15
	movq	384(%rsp), %r12
	movq	392(%rsp), %r14
	movzbl	400(%rsp), %ebp
	movzbl	360(%rsp), %r13d
	movabsq	$4611686018427387904, %rax
	testq	%rax, 648(%rsp)
	je	.LBB0_89
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_89
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	jne	.LBB0_90
	jmp	.LBB0_88
.LBB0_89:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	je	.LBB0_88
.LBB0_90:
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r12
	je	.LBB0_93
	lock		decq	-8(%r15)
	jne	.LBB0_93
	addq	$-8, %r15
	#MEMBARRIER
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_93:
	movl	$1, %ebx
	cmpb	$1, %bpl
	je	.LBB0_17
	jmp	.LBB0_82
.LBB0_94:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
.LBB0_88:
	xorl	%ebx, %ebx
	jmp	.LBB0_82
.LBB0_96:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	632(%rsp), %r8
	movl	$783, %edi
	movq	$35, 640(%rsp)
	jmp	.LBB0_97
.LBB0_98:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movq	$37, 640(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	632(%rsp), %r8
	movl	$659, %edi
.LBB0_97:
	movq	%rax, 632(%rsp)
	movq	%r13, 648(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end0:
	.size	gate_wmma, .Lfunc_end0-gate_wmma
	.size	.Lgate_wmma$local, .Lfunc_end0-gate_wmma
	.cfi_endproc

	.globl	gate_loadtr
	.prefalign	4, .Lfunc_end1, nop
	.type	gate_loadtr,@function
gate_loadtr:
.Lgate_loadtr$local:
	.type	.Lgate_loadtr$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4728, %rsp
	.cfi_def_cfa_offset 4784
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 16(%rsp)
	leaq	static_string_3b45d8d0deb8f5b3(%rip), %rax
	leaq	208(%rsp), %rbx
	movl	$4, %esi
	movq	%rdi, 192(%rsp)
	movabsq	$2305843009213693952, %rbp
	movq	$0, 48(%rsp)
	movq	%rax, 208(%rsp)
	movq	$3, 216(%rsp)
	movq	%rbx, %rdi
	movq	$0, 224(%rsp)
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB134_2
	movq	216(%rsp), %rcx
	jmp	.LBB134_3
.LBB134_2:
	movl	$1336, %edx
	bextrq	%rdx, %rcx, %rcx
.LBB134_3:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	movq	%rax, %rcx
	orq	%rbp, %rcx
	movq	%rcx, 224(%rsp)
	testq	%rax, %rax
	js	.LBB134_5
	movq	208(%rsp), %rbx
.LBB134_5:
	leaq	48(%rsp), %rdi
	movq	%rbx, %rsi
	xorl	%edx, %edx
	callq	AsyncRT_DeviceContext_create@PLT
	testb	$64, 231(%rsp)
	je	.LBB134_8
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_8
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %rbx
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rax
.LBB134_8:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movabsq	$4611686018427387904, %r15
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB134_18
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_69b8ed51c73424d7(%rip), %r9
	leaq	536(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$4010, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$41
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	%r15, %rax
	movq	536(%rsp), %rbx
	movq	552(%rsp), %r13
	movq	560(%rsp), %r14
	movzbl	568(%rsp), %r12d
	movzbl	528(%rsp), %r15d
	movq	%rax, %rbp
	testq	%rax, 632(%rsp)
	je	.LBB134_12
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_12
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_12:
	testb	$1, %r15b
	movq	%rbp, %r15
	movabsq	$2305843009213693952, %rbp
	je	.LBB134_18
	testq	%r15, %r13
	je	.LBB134_16
	lock		decq	-8(%rbx)
	jne	.LBB134_16
	addq	$-8, %rbx
	#MEMBARRIER
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_16:
	movl	$1, %ebx
	cmpb	$1, %r12b
	jne	.LBB134_84
	movq	%r14, %rdi
	jmp	.LBB134_83
.LBB134_18:
	movq	48(%rsp), %rbx
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_retain@PLT
	xorl	%edi, %edi
	movq	$0, 48(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	cmpq	$23, %rax
	jg	.LBB134_22
	leaq	208(%rsp), %rdi
	movabsq	$-9223372036854775808, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, 208(%rsp)
	movq	%rax, 224(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movq	224(%rsp), %rax
	testq	%rbp, %rax
	jne	.LBB134_20
.LBB134_27:
	movl	$1336, %r14d
	testq	%rax, %rax
	js	.LBB134_29
	movq	216(%rsp), %rsi
	jmp	.LBB134_30
.LBB134_22:
	movq	%rax, %r14
	addq	$7, %r14
	movq	%r14, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB134_98
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB134_100
	sarq	$3, %r14
	movq	$1, (%rax)
	addq	$8, %rax
	movq	$0, 4712(%rsp)
	orq	%r15, %r14
	movq	%rax, 208(%rsp)
	movq	$0, 216(%rsp)
	leaq	208(%rsp), %rax
	movq	%r14, 224(%rsp)
	leaq	616(%rsp), %r14
	movq	%rax, 4720(%rsp)
	movq	%r14, %rdi
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4712(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB134_26
	movq	4720(%rsp), %rdi
	leaq	616(%rsp), %r14
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4712(%rsp)
.LBB134_26:
	movq	4720(%rsp), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	224(%rsp), %rax
	testq	%rbp, %rax
	je	.LBB134_27
.LBB134_20:
	testq	%rax, %rax
	js	.LBB134_34
.LBB134_21:
	movq	208(%rsp), %rax
	jmp	.LBB134_35
.LBB134_29:
	bextrq	%r14, %rax, %rsi
.LBB134_30:
	incq	%rsi
	leaq	208(%rsp), %rdi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB134_32
	movq	216(%rsp), %rcx
	jmp	.LBB134_33
.LBB134_32:
	bextrq	%r14, %rcx, %rcx
.LBB134_33:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	orq	%rbp, %rax
	movq	%rax, 224(%rsp)
	testq	%rax, %rax
	jns	.LBB134_21
.LBB134_34:
	leaq	208(%rsp), %rax
.LBB134_35:
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_49fbd36d4f006cd8(%rip), %rdx
	leaq	static_string_822c3714e12be7aa(%rip), %rcx
	leaq	static_string_e64fd374c68b723d(%rip), %r8
	leaq	56(%rsp), %rdi
	movl	$3504, %r9d
	movq	%rbx, %rsi
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$-1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_loadFunction@PLT
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	testq	%r15, 224(%rsp)
	je	.LBB134_38
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_38
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r14, %rax
.LBB134_38:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB134_45
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f6c401c515dc2a68(%rip), %r9
	leaq	488(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$170, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$49
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	488(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	512(%rsp), %rax
	movq	504(%rsp), %r13
	movq	%rax, 88(%rsp)
	movq	%r15, %rax
	movq	%rax, %r14
	movzbl	520(%rsp), %r12d
	movzbl	480(%rsp), %r15d
	testq	%rax, 632(%rsp)
	je	.LBB134_43
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_43
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testb	$1, %r15b
	jne	.LBB134_44
.LBB134_42:
	movq	%r14, %r15
	movq	48(%rsp), %r14
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	jne	.LBB134_46
.LBB134_49:
	leaq	592(%rsp), %rcx
	leaq	599(%rsp), %rax
	movq	%r14, 8(%rsp)
	leaq	616(%rsp), %r13
	movq	$1, 616(%rsp)
	movq	$0, 624(%rsp)
	testq	%rcx, %rcx
	movq	%r13, %rdi
	cmovnsq	%rcx, %rax
	setns	%dl
	sarq	$3, %rax
	movq	%rax, %rsi
	negq	%rsi
	shlq	$3, %rsi
	addq	%rcx, %rsi
	movq	%rbx, %rsi
	setne	%cl
	andb	%dl, %cl
	movzbl	%cl, %r14d
	addq	%rax, %r14
	shlq	$3, %r14
	leaq	8(%r14), %r12
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	movq	624(%rsp), %rdx
	cmpq	$5, %rdx
	jne	.LBB134_54
	movq	616(%rsp), %rax
	leaq	static_string_561d9efdd15f277f(%rip), %rcx
	cmpq	%rcx, %rax
	je	.LBB134_56
	movq	%rdx, %rsi
	sarq	$63, %rsi
	andnq	%rdx, %rsi, %rdx
	xorl	%esi, %esi
	.p2align	4
.LBB134_52:
	cmpq	%rsi, %rdx
	je	.LBB134_56
	movzbl	(%rax,%rsi), %edi
	cmpb	(%rsi,%rcx), %dil
	leaq	1(%rsi), %rsi
	je	.LBB134_52
.LBB134_54:
	movq	192(%rsp), %rax
	movq	16(%rsp), %rsi
	movq	%r14, 208(%rsp)
	movq	%r12, 216(%rsp)
	movq	$0, 576(%rsp)
	movq	%rbx, %rdi
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%rax, (%r14)
	movq	%rsi, 8(%r14)
	movq	8(%rsp), %r14
	leaq	208(%rsp), %rax
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$2
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB134_85
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	616(%rsp), %r13
	movq	%rax, 96(%rsp)
	movq	%rdx, 104(%rsp)
	movq	$1, 616(%rsp)
	movq	%rcx, 112(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbx, %rsi
	movq	%r13, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	616(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 168(%rsp)
	movq	$0, 184(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 128(%rsp)
	movq	$12, 152(%rsp)
	movq	$1, 32(%rsp)
	movq	$13, 56(%rsp)
	movq	$1, 624(%rsp)
	movq	%rcx, 120(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 136(%rsp)
	movq	%rcx, 144(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 160(%rsp)
	movq	%rcx, 24(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 40(%rsp)
	movq	%rcx, 48(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 64(%rsp)
	movq	%rcx, 616(%rsp)
	movq	%rbp, 632(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	152(%rsp), %r12
	leaq	static_string_f1dde65067a8a2a0(%rip), %rbp
	leaq	104(%rsp), %r10
	leaq	32(%rsp), %r11
	leaq	176(%rsp), %r14
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	448(%rsp), %rdi
	leaq	128(%rsp), %r9
	movl	$103, %esi
	movl	$44, %edx
	movl	$25, %r8d
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	leaq	72(%rsp), %r10
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$624
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movq	464(%rsp), %rax
	movzbl	472(%rsp), %ebp
	movq	%rax, 88(%rsp)
	movq	456(%rsp), %r12
	movq	440(%rsp), %r13
	testq	%r15, 632(%rsp)
	jne	.LBB134_58
	jmp	.LBB134_60
.LBB134_43:
	testb	$1, %r15b
	je	.LBB134_42
.LBB134_44:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %r15
	jmp	.LBB134_47
.LBB134_45:
	movq	48(%rsp), %r14
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	je	.LBB134_49
.LBB134_46:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
.LBB134_47:
	movl	%r12d, %ebp
	movq	%r13, %r12
	movq	8(%rsp), %r13
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %r12
	jne	.LBB134_79
	jmp	.LBB134_81
.LBB134_56:
	movq	192(%rsp), %rax
	movq	16(%rsp), %rsi
	movq	%r14, 208(%rsp)
	movq	%r12, 216(%rsp)
	leaq	48(%rsp), %r12
	vxorps	%xmm0, %xmm0, %xmm0
	vmovups	%xmm0, 26(%rsp)
	movb	$0, 24(%rsp)
	vmovups	%zmm0, 696(%rsp)
	vmovups	%zmm0, 632(%rsp)
	movq	$8, 616(%rsp)
	movq	$8, 624(%rsp)
	movb	$0, 25(%rsp)
	movq	$0, 584(%rsp)
	movq	%rbx, %rdi
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%r12, 200(%rsp)
	movq	%rax, (%r14)
	movq	%rsi, 8(%r14)
	movq	8(%rsp), %r14
	leaq	208(%rsp), %rax
	movq	%rax, 48(%rsp)
	movq	%r13, 56(%rsp)
	leaq	24(%rsp), %r13
	leaq	200(%rsp), %rax
	movq	%r13, 64(%rsp)
	movq	$8, 72(%rsp)
	movl	$0, 80(%rsp)
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$2
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	vzeroupper
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB134_85
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	616(%rsp), %r14
	movq	%rax, 96(%rsp)
	movq	%rdx, 104(%rsp)
	movq	$1, 616(%rsp)
	movq	%rcx, 112(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	616(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 168(%rsp)
	movq	$0, 184(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 128(%rsp)
	movq	$12, 152(%rsp)
	movq	$1, 32(%rsp)
	movq	$13, 56(%rsp)
	movq	$1, 624(%rsp)
	movq	%rcx, 120(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 136(%rsp)
	movq	%rcx, 144(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 160(%rsp)
	movq	%rcx, 24(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 40(%rsp)
	movq	%rcx, 48(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 64(%rsp)
	movq	%rcx, 616(%rsp)
	movq	%rbp, 632(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movq	%r12, %rbp
	leaq	152(%rsp), %r12
	leaq	104(%rsp), %r10
	leaq	176(%rsp), %r11
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	408(%rsp), %rdi
	leaq	128(%rsp), %r9
	movl	$103, %esi
	movl	$44, %edx
	movl	$25, %r8d
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$624
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f1dde65067a8a2a0(%rip), %rax
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movq	424(%rsp), %rax
	movzbl	432(%rsp), %ebp
	movq	%rax, 88(%rsp)
	movq	416(%rsp), %r12
	movq	400(%rsp), %r13
	testq	%r15, 632(%rsp)
	je	.LBB134_60
.LBB134_58:
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_60
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_60:
	testq	%r15, 64(%rsp)
	je	.LBB134_63
	movq	48(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_63
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_63:
	movq	8(%rsp), %r14
	testq	%r15, 40(%rsp)
	je	.LBB134_66
	movq	24(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_66
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_66:
	testq	%r15, 160(%rsp)
	je	.LBB134_69
	movq	144(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_69
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_69:
	testq	%r15, 136(%rsp)
	je	.LBB134_72
	movq	120(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_72
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_72:
	testq	%r15, 184(%rsp)
	je	.LBB134_75
	movq	168(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_75
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_75:
	testq	%r15, 112(%rsp)
	je	.LBB134_78
	movq	96(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_78
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_78:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %r12
	je	.LBB134_81
.LBB134_79:
	lock		decq	-8(%r13)
	jne	.LBB134_81
	addq	$-8, %r13
	#MEMBARRIER
	movq	%r13, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_81:
	movl	$1, %ebx
	cmpb	$1, %bpl
	jne	.LBB134_84
	movq	88(%rsp), %rdi
.LBB134_83:
	callq	"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]"@PLT
.LBB134_84:
	movq	%rbx, %rax
	addq	$4728, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB134_85:
	.cfi_def_cfa_offset 4784
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_synchronize@PLT
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB134_96
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f078bd8d2bcbf530(%rip), %r9
	leaq	360(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$104, %ecx
	movl	$24, %r8d
	movq	%rax, %rsi
	pushq	$25
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	384(%rsp), %rcx
	movq	%r15, %rax
	movq	360(%rsp), %r15
	movq	376(%rsp), %r12
	movq	%rax, %r14
	movq	%rcx, 16(%rsp)
	movzbl	392(%rsp), %ebp
	movzbl	352(%rsp), %r13d
	testq	%rax, 632(%rsp)
	je	.LBB134_90
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB134_90
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	jne	.LBB134_91
	jmp	.LBB134_89
.LBB134_90:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	je	.LBB134_89
.LBB134_91:
	testq	%r14, %r12
	je	.LBB134_94
	lock		decq	-8(%r15)
	jne	.LBB134_94
	addq	$-8, %r15
	#MEMBARRIER
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB134_94:
	movl	$1, %ebx
	cmpb	$1, %bpl
	jne	.LBB134_84
	movq	16(%rsp), %rdi
	jmp	.LBB134_83
.LBB134_96:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
.LBB134_89:
	xorl	%ebx, %ebx
	jmp	.LBB134_84
.LBB134_98:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	616(%rsp), %r8
	movl	$783, %edi
	movq	$35, 624(%rsp)
	jmp	.LBB134_99
.LBB134_100:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movq	$37, 624(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	616(%rsp), %r8
	movl	$659, %edi
.LBB134_99:
	movq	%rax, 616(%rsp)
	movq	%rbp, 632(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end1:
	.size	gate_loadtr, .Lfunc_end1-gate_loadtr
	.size	.Lgate_loadtr$local, .Lfunc_end1-gate_loadtr
	.cfi_endproc

	.prefalign	4, .Lfunc_end2, nop
	.type	"std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])",@function
"std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])":
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	testq	%rdi, %rdi
	js	.LBB135_1
	movq	%rdi, %rsi
	leaq	8(%rsp), %rdi
	movq	$0, 8(%rsp)
	callq	KGEN_CompilerRT_GetStackTrace@PLT
	testq	%rax, %rax
	je	.LBB135_1
	movq	8(%rsp), %rbx
	movl	$8, %edi
	movl	$24, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB135_6
	movb	$1, %dl
	movq	$1, (%rax)
	movq	$1, 8(%rax)
	movq	%rbx, 16(%rax)
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB135_1:
	.cfi_def_cfa_offset 48
	xorl	%edx, %edx
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB135_6:
	.cfi_def_cfa_offset 48
	leaq	static_string_09e773a88105e290(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	movq	$37, 16(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	8(%rsp), %r8
	movl	$659, %edi
	movl	$14, %esi
	movq	%rax, 8(%rsp)
	movq	%rcx, 24(%rsp)
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end2:
	.size	"std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])", .Lfunc_end2-"std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])"
	.cfi_endproc

	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
	.type	.LCPI136_0,@object
.LCPI136_0:
	.quad	31
	.size	.LCPI136_0, 8
	.text
	.prefalign	4, .Lfunc_end3, nop
	.type	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]",@function
"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4184, %rsp
	.cfi_def_cfa_offset 4240
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, %r14
	movq	%rsi, %rbp
	movq	4296(%rsp), %rsi
	movq	4288(%rsp), %r9
	movq	%rcx, 56(%rsp)
	movq	%rdx, 32(%rsp)
	movq	4304(%rsp), %r10
	movl	$1336, %eax
	movq	16(%r9), %rcx
	movq	16(%rsi), %rdx
	testq	%rcx, %rcx
	js	.LBB136_4
	movq	8(%r9), %r12
	movq	16(%r10), %rcx
	movq	%rdi, 24(%rsp)
	testq	%rdx, %rdx
	jns	.LBB136_5
.LBB136_2:
	bextrq	%rax, %rdx, %r15
	leaq	1(%r8), %rsi
	movq	%r8, 64(%rsp)
	testq	%rcx, %rcx
	jns	.LBB136_6
.LBB136_3:
	bextrq	%rax, %rcx, %r13
	jmp	.LBB136_7
.LBB136_4:
	bextrq	%rax, %rcx, %r12
	movq	16(%r10), %rcx
	movq	%rdi, 24(%rsp)
	testq	%rdx, %rdx
	js	.LBB136_2
.LBB136_5:
	movq	8(%rsi), %r15
	leaq	1(%r8), %rsi
	movq	%r8, 64(%rsp)
	testq	%rcx, %rcx
	js	.LBB136_3
.LBB136_6:
	movq	8(%r10), %r13
.LBB136_7:
	movq	%rbp, %rdi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	movq	32(%rsp), %rbx
	leaq	1(%rax), %rsi
	movq	%rbx, %rdi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	movq	4256(%rsp), %rcx
	movq	4272(%rsp), %rsi
	movq	4264(%rsp), %rdx
	vmovdqu	8(%r14), %xmm0
	movq	4280(%rsp), %rdi
	vmovdqu	8(%rcx), %xmm1
	vmovdqu	8(%rsi), %xmm2
	vpbroadcastq	16(%rdx), %xmm4
	movq	4248(%rsp), %rcx
	addq	%rcx, %rax
	vpunpckhqdq	%xmm1, %xmm0, %xmm3
	vinserti128	$1, %xmm4, %ymm3, %ymm3
	vinserti128	$1, %xmm2, %ymm0, %ymm4
	vpunpcklqdq	%xmm1, %xmm0, %xmm0
	vinserti128	$1, 8(%rdx), %ymm0, %ymm0
	vpbroadcastq	%xmm2, %ymm1
	vpblendd	$192, %ymm4, %ymm3, %ymm3
	vpsrlq	$56, %ymm3, %ymm4
	vpmovq2m	%ymm3, %k1
	vpblendd	$192, %ymm1, %ymm0, %ymm0
	vpandq	.LCPI136_0(%rip){1to4}, %ymm4, %ymm0 {%k1}
	vextracti128	$1, %ymm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vpshufd	$238, %xmm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vmovq	%xmm0, %rsi
	addq	%rax, %rsi
	vzeroupper
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	addq	%r12, %r15
	addq	%r13, %r15
	addq	%rax, %r15
	cmpq	$23, %r15
	jg	.LBB136_10
	movq	56(%rsp), %rdx
	movq	64(%rsp), %rcx
	movabsq	$-9223372036854775808, %rax
	vpxor	%xmm0, %xmm0, %xmm0
	vmovdqa	%xmm0, (%rsp)
	movq	%rsp, %r8
	movq	%rbp, %rdi
	movq	%rbx, %rsi
	movq	%rax, 16(%rsp)
	callq	"std::reflection::location::SourceLocation::write_to[::Writer & ::AnyType](::SourceLocation,$0),writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movq	16(%r14), %rax
	testq	%rax, %rax
	js	.LBB136_20
	movq	8(%r14), %rdx
	movq	(%r14), %r14
	jmp	.LBB136_21
.LBB136_10:
	addq	$7, %r15
	movq	%rbp, 48(%rsp)
	movq	%r15, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB136_78
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB136_80
	movq	56(%rsp), %rsi
	movq	64(%rsp), %rdx
	movq	$1, (%rax)
	addq	$8, %rax
	sarq	$3, %r15
	leaq	72(%rsp), %rdi
	movq	$0, 4168(%rsp)
	movq	%rax, (%rsp)
	movabsq	$4611686018427387904, %rax
	movq	$0, 8(%rsp)
	orq	%r15, %rax
	movq	%rax, 16(%rsp)
	movq	%rsp, %rax
	movq	%rax, 4176(%rsp)
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	leaq	1(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB136_14
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4168(%rsp)
.LBB136_14:
	movb	$58, 72(%rsp,%rdx)
	movq	48(%rsp), %rdi
	incq	4168(%rsp)
	movq	24(%rsp), %r15
	movq	4272(%rsp), %r12
	movq	4264(%rsp), %r13
	movq	4240(%rsp), %rbp
	movq	4256(%rsp), %rbx
	leaq	72(%rsp), %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4168(%rsp), %rdx
	leaq	1(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB136_16
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4168(%rsp)
.LBB136_16:
	movb	$58, 72(%rsp,%rdx)
	movq	32(%rsp), %rdi
	incq	4168(%rsp)
	leaq	72(%rsp), %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_18
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_18:
	movq	16(%r14), %rax
	testq	%rax, %rax
	js	.LBB136_39
	movq	8(%r14), %rdx
	movq	(%r14), %r14
	jmp	.LBB136_40
.LBB136_20:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_21:
	movq	4272(%rsp), %r12
	movq	4264(%rsp), %r13
	movq	4240(%rsp), %rbp
	movq	4256(%rsp), %rbx
	movq	%rsp, %r15
	movq	%r15, %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	4248(%rsp), %rdx
	movq	%r15, %rdi
	movq	%rbp, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	16(%rbx), %rax
	testq	%rax, %rax
	js	.LBB136_23
	movq	8(%rbx), %rdx
	movq	(%rbx), %rbx
	jmp	.LBB136_24
.LBB136_23:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_24:
	movq	24(%rsp), %r15
	movq	%rsp, %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	16(%r13), %rax
	testq	%rax, %rax
	js	.LBB136_26
	movq	8(%r13), %rdx
	movq	(%r13), %r13
	jmp	.LBB136_27
.LBB136_26:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_27:
	movq	%rsp, %rdi
	movq	%r13, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	16(%r12), %rax
	testq	%rax, %rax
	js	.LBB136_29
	movq	8(%r12), %rdx
	movq	(%r12), %r12
	jmp	.LBB136_30
.LBB136_29:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_30:
	movq	%rsp, %rbx
	movq	%rbx, %rdi
	movq	%r12, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	4280(%rsp), %rdi
	movq	%rbx, %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movq	4288(%rsp), %rsi
	movq	16(%rsi), %rax
	testq	%rax, %rax
	js	.LBB136_32
	movq	8(%rsi), %rdx
	movq	(%rsi), %rsi
	jmp	.LBB136_33
.LBB136_32:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_33:
	movq	%rsp, %rdi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	4296(%rsp), %rsi
	movq	16(%rsi), %rax
	testq	%rax, %rax
	js	.LBB136_35
	movq	8(%rsi), %rdx
	movq	(%rsi), %rsi
	jmp	.LBB136_36
.LBB136_35:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_36:
	movq	%rsp, %rdi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	4304(%rsp), %rsi
	movq	16(%rsi), %rax
	testq	%rax, %rax
	js	.LBB136_38
	movq	8(%rsi), %rdx
	movq	(%rsi), %rsi
	movq	%rsp, %rdi
	jmp	.LBB136_77
.LBB136_38:
	movl	$1336, %ecx
	movq	%rsp, %rdi
	bextrq	%rcx, %rax, %rdx
	jmp	.LBB136_77
.LBB136_39:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_40:
	leaq	72(%rsp), %rdi
	movq	%r14, %rsi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_42
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_42:
	movq	4248(%rsp), %rdx
	leaq	72(%rsp), %rdi
	movq	%rbp, %rsi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_44
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_44:
	movq	16(%rbx), %rax
	testq	%rax, %rax
	js	.LBB136_46
	movq	8(%rbx), %rdx
	movq	(%rbx), %rbx
	jmp	.LBB136_47
.LBB136_46:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_47:
	leaq	72(%rsp), %rdi
	movq	%rbx, %rsi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_49
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_49:
	movq	16(%r13), %rax
	testq	%rax, %rax
	js	.LBB136_51
	movq	8(%r13), %rdx
	movq	(%r13), %r13
	jmp	.LBB136_52
.LBB136_51:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_52:
	leaq	72(%rsp), %rdi
	movq	%r13, %rsi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_54
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_54:
	movq	16(%r12), %rax
	testq	%rax, %rax
	js	.LBB136_56
	movq	8(%r12), %rdx
	movq	(%r12), %r12
	jmp	.LBB136_57
.LBB136_56:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_57:
	leaq	72(%rsp), %rdi
	movq	%r12, %rsi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_59
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_59:
	movq	4280(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_61
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_61:
	movq	4288(%rsp), %rsi
	movq	16(%rsi), %rax
	testq	%rax, %rax
	js	.LBB136_63
	movq	8(%rsi), %rdx
	movq	(%rsi), %rsi
	jmp	.LBB136_64
.LBB136_63:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_64:
	leaq	72(%rsp), %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_66
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_66:
	movq	4296(%rsp), %rsi
	movq	16(%rsi), %rax
	testq	%rax, %rax
	js	.LBB136_68
	movq	8(%rsi), %rdx
	movq	(%rsi), %rsi
	jmp	.LBB136_69
.LBB136_68:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_69:
	leaq	72(%rsp), %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB136_71
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4168(%rsp)
.LBB136_71:
	movq	4304(%rsp), %rsi
	movq	16(%rsi), %rax
	testq	%rax, %rax
	js	.LBB136_73
	movq	8(%rsi), %rdx
	movq	(%rsi), %rsi
	jmp	.LBB136_74
.LBB136_73:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB136_74:
	leaq	72(%rsp), %rbx
	movq	%rbx, %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4168(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB136_76
	movq	4176(%rsp), %rdi
	leaq	72(%rsp), %rbx
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4168(%rsp)
.LBB136_76:
	movq	4176(%rsp), %rdi
	movq	%rbx, %rsi
.LBB136_77:
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	vmovups	8(%rsp), %xmm0
	movq	(%rsp), %rbx
	xorl	%edi, %edi
	vmovaps	%xmm0, 32(%rsp)
	callq	"std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])"@PLT
	movb	%dl, 32(%r15)
	movq	%rax, 24(%r15)
	movq	%r15, %rax
	vmovaps	32(%rsp), %xmm0
	vmovups	%xmm0, 8(%r15)
	movq	%rbx, (%r15)
	addq	$4184, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB136_78:
	.cfi_def_cfa_offset 4240
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	72(%rsp), %r8
	movl	$783, %edi
	movq	$35, 80(%rsp)
	jmp	.LBB136_79
.LBB136_80:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	movq	$37, 80(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	72(%rsp), %r8
	movl	$659, %edi
.LBB136_79:
	movq	%rax, 72(%rsp)
	movq	%rcx, 88(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end3:
	.size	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]", .Lfunc_end3-"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end4, nop
	.type	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]",@function
"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$4112, %rsp
	.cfi_def_cfa_offset 4144
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	16(%rdi), %rcx
	movq	%rdx, %rax
	testq	%rcx, %rcx
	js	.LBB137_7
	movq	8(%rdi), %r8
	addq	%r8, %rax
	cmpq	$24, %rax
	jl	.LBB137_8
	movq	%rcx, %r9
	shrq	$62, %r9
	shlq	$3, %rcx
	testq	%r9, %r9
	cmoveq	%r8, %rcx
	cmpq	%rcx, %rax
	jg	.LBB137_3
	jmp	.LBB137_4
.LBB137_7:
	movl	$1336, %r8d
	bextrq	%r8, %rcx, %rcx
	addq	%rcx, %rax
	cmpq	$23, %rax
	jg	.LBB137_3
.LBB137_8:
	addq	$4112, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.LBB137_3:
	.cfi_def_cfa_offset 4144
	movq	%rsi, %r14
	movq	%rax, %rsi
	movq	%rdi, %rbx
	movq	%rdx, %r15
	callq	"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])"@PLT
	movq	%rbx, %rdi
	movq	%r15, %rdx
	movq	%r14, %rsi
.LBB137_4:
	movq	$0, 4096(%rsp)
	movq	%rdi, 4104(%rsp)
	movq	%rsp, %rbx
	movq	%rbx, %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4096(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB137_6
	movq	4104(%rsp), %rdi
	movq	%rsp, %rbx
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rsp)
.LBB137_6:
	movq	4104(%rsp), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	addq	$4112, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]", .Lfunc_end4-"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end5, nop
	.type	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=true,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]",@function
"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=true,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$4112, %rsp
	.cfi_def_cfa_offset 4144
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	16(%rdi), %rcx
	movq	%rdx, %rax
	testq	%rcx, %rcx
	js	.LBB138_7
	movq	8(%rdi), %r8
	addq	%r8, %rax
	cmpq	$24, %rax
	jl	.LBB138_8
	movq	%rcx, %r9
	shrq	$62, %r9
	shlq	$3, %rcx
	testq	%r9, %r9
	cmoveq	%r8, %rcx
	cmpq	%rcx, %rax
	jg	.LBB138_3
	jmp	.LBB138_4
.LBB138_7:
	movl	$1336, %r8d
	bextrq	%r8, %rcx, %rcx
	addq	%rcx, %rax
	cmpq	$23, %rax
	jg	.LBB138_3
.LBB138_8:
	addq	$4112, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.LBB138_3:
	.cfi_def_cfa_offset 4144
	movq	%rsi, %r14
	movq	%rax, %rsi
	movq	%rdi, %rbx
	movq	%rdx, %r15
	callq	"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])"@PLT
	movq	%rbx, %rdi
	movq	%r15, %rdx
	movq	%r14, %rsi
.LBB138_4:
	movq	$0, 4096(%rsp)
	movq	%rdi, 4104(%rsp)
	movq	%rsp, %rbx
	movq	%rbx, %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=true"@PLT
	movq	4096(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB138_6
	movq	4104(%rsp), %rdi
	movq	%rsp, %rbx
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rsp)
.LBB138_6:
	movq	4104(%rsp), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	addq	$4112, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end5:
	.size	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=true,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]", .Lfunc_end5-"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=true,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end6, nop
	.type	"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[DType.uint8, 1], $2, AddressSpace.GENERIC],::Span[$3, $4, ::SIMD[DType.uint8, 1], $5, AddressSpace.GENERIC]),lhs.mut`2x=false,rhs.mut`2x3=false",@function
"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[DType.uint8, 1], $2, AddressSpace.GENERIC],::Span[$3, $4, ::SIMD[DType.uint8, 1], $5, AddressSpace.GENERIC]),lhs.mut`2x=false,rhs.mut`2x3=false":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leaq	(%rcx,%rsi), %r13
	movq	%rcx, %rbx
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %r12
	cmpq	$23, %r13
	jg	.LBB139_2
	movabsq	$9223372036854775776, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, (%rsp)
	addq	$32, %rax
	jmp	.LBB139_6
.LBB139_2:
	leaq	7(%r13), %rbp
	movq	%rbp, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB139_34
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB139_36
	movq	$1, (%rax)
	addq	$8, %rax
	sarq	$3, %rbp
	movq	%rax, (%rsp)
	movabsq	$4611686018427387904, %rax
	movq	$0, 8(%rsp)
	orq	%rbp, %rax
	movq	%rax, 16(%rsp)
	testq	%rbp, %rbp
	js	.LBB139_6
	movq	%r13, 8(%rsp)
	jmp	.LBB139_7
.LBB139_6:
	movabsq	$-2233785415175766017, %rcx
	shlq	$56, %r13
	andq	%rax, %rcx
	orq	%r13, %rcx
	movq	%rcx, 16(%rsp)
.LBB139_7:
	movq	%rsp, %rdi
	xorl	%esi, %esi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	%r15, %rcx
	subq	$1, %rcx
	jae	.LBB139_10
.LBB139_8:
	movq	%rbx, %rcx
	subq	$1, %rcx
	jae	.LBB139_13
.LBB139_9:
	movq	(%rsp), %rax
	movq	8(%rsp), %rdx
	movq	16(%rsp), %rcx
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB139_10:
	.cfi_def_cfa_offset 112
	cmpq	$4, %r15
	jg	.LBB139_16
	movzbl	(%r12), %edx
	movb	%dl, (%rax)
	movzbl	(%r12,%rcx), %edx
	movb	%dl, (%rax,%rcx)
	cmpq	$3, %r15
	jl	.LBB139_8
	movzbl	1(%r12), %ecx
	movb	%cl, 1(%rax)
	movzbl	-2(%r12,%r15), %ecx
	movb	%cl, -2(%rax,%r15)
	jmp	.LBB139_8
.LBB139_13:
	addq	%r15, %rax
	cmpq	$4, %rbx
	jg	.LBB139_19
	movzbl	(%r14), %edx
	movb	%dl, (%rax)
	movzbl	(%r14,%rcx), %edx
	movb	%dl, (%rax,%rcx)
	cmpq	$3, %rbx
	jl	.LBB139_9
	movzbl	1(%r14), %ecx
	movb	%cl, 1(%rax)
	movzbl	-2(%r14,%rbx), %ecx
	movb	%cl, -2(%rax,%rbx)
	jmp	.LBB139_9
.LBB139_16:
	cmpq	$16, %r15
	ja	.LBB139_22
	cmpq	$8, %r15
	jb	.LBB139_32
	movq	(%r12), %rcx
	movq	%rcx, (%rax)
	movq	-8(%r12,%r15), %rcx
	movq	%rcx, -8(%rax,%r15)
	jmp	.LBB139_8
.LBB139_19:
	cmpq	$16, %rbx
	ja	.LBB139_27
	cmpq	$8, %rbx
	jb	.LBB139_33
	movq	(%r14), %rcx
	movq	%rcx, (%rax)
	movq	-8(%r14,%rbx), %rcx
	movq	%rcx, -8(%rax,%rbx)
	jmp	.LBB139_9
.LBB139_22:
	movabsq	$9223372036854775776, %rcx
	andq	%r15, %rcx
	je	.LBB139_25
	xorl	%edx, %edx
	.p2align	4
.LBB139_24:
	vmovups	(%r12,%rdx), %ymm0
	vmovups	%ymm0, (%rax,%rdx)
	addq	$32, %rdx
	cmpq	%rcx, %rdx
	jb	.LBB139_24
.LBB139_25:
	cmpq	%r15, %rcx
	je	.LBB139_8
	.p2align	4
.LBB139_26:
	movzbl	(%r12,%rcx), %edx
	movb	%dl, (%rax,%rcx)
	incq	%rcx
	cmpq	%rcx, %r15
	jne	.LBB139_26
	jmp	.LBB139_8
.LBB139_27:
	movabsq	$9223372036854775776, %rcx
	andq	%rbx, %rcx
	je	.LBB139_30
	xorl	%edx, %edx
	.p2align	4
.LBB139_29:
	vmovups	(%r14,%rdx), %ymm0
	vmovups	%ymm0, (%rax,%rdx)
	addq	$32, %rdx
	cmpq	%rcx, %rdx
	jb	.LBB139_29
.LBB139_30:
	cmpq	%rbx, %rcx
	je	.LBB139_9
	.p2align	4
.LBB139_31:
	movzbl	(%r14,%rcx), %edx
	movb	%dl, (%rax,%rcx)
	incq	%rcx
	cmpq	%rcx, %rbx
	jne	.LBB139_31
	jmp	.LBB139_9
.LBB139_32:
	movl	(%r12), %ecx
	movl	%ecx, (%rax)
	movl	-4(%r12,%r15), %ecx
	movl	%ecx, -4(%rax,%r15)
	jmp	.LBB139_8
.LBB139_33:
	movl	(%r14), %ecx
	movl	%ecx, (%rax)
	movl	-4(%r14,%rbx), %ecx
	movl	%ecx, -4(%rax,%rbx)
	jmp	.LBB139_9
.LBB139_34:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	32(%rsp), %r8
	movl	$783, %edi
	movq	$35, 40(%rsp)
	jmp	.LBB139_35
.LBB139_36:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	movq	$37, 40(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	32(%rsp), %r8
	movl	$659, %edi
.LBB139_35:
	movq	%rax, 32(%rsp)
	movq	%rcx, 48(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end6:
	.size	"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[DType.uint8, 1], $2, AddressSpace.GENERIC],::Span[$3, $4, ::SIMD[DType.uint8, 1], $5, AddressSpace.GENERIC]),lhs.mut`2x=false,rhs.mut`2x3=false", .Lfunc_end6-"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[DType.uint8, 1], $2, AddressSpace.GENERIC],::Span[$3, $4, ::SIMD[DType.uint8, 1], $5, AddressSpace.GENERIC]),lhs.mut`2x=false,rhs.mut`2x3=false"
	.cfi_endproc

	.prefalign	4, .Lfunc_end7, nop
	.type	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])",@function
"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r13
	subq	$1, %r13
	jb	.LBB140_20
	movq	16(%rdi), %rax
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
	testq	%rax, %rax
	js	.LBB140_2
	movq	8(%rbx), %rbp
	jmp	.LBB140_4
.LBB140_2:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rbp
.LBB140_4:
	leaq	(%rbp,%r14), %r12
	movq	%rbx, %rdi
	movq	%r12, %rsi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	addq	%rbp, %rax
	cmpq	$4, %r14
	jg	.LBB140_7
	movzbl	(%r15), %ecx
	movb	%cl, (%rax)
	movzbl	(%r15,%r13), %ecx
	movb	%cl, (%rax,%r13)
	cmpq	$3, %r14
	jl	.LBB140_16
	movzbl	1(%r15), %ecx
	movb	%cl, 1(%rax)
	movzbl	-2(%r15,%r14), %ecx
	movb	%cl, -2(%rax,%r14)
	movq	16(%rbx), %rax
	testq	%rax, %rax
	jns	.LBB140_18
	jmp	.LBB140_17
.LBB140_7:
	cmpq	$16, %r14
	ja	.LBB140_11
	cmpq	$8, %r14
	jb	.LBB140_10
	movq	(%r15), %rcx
	movq	%rcx, (%rax)
	movq	-8(%r15,%r14), %rcx
	movq	%rcx, -8(%rax,%r14)
	movq	16(%rbx), %rax
	testq	%rax, %rax
	jns	.LBB140_18
	jmp	.LBB140_17
.LBB140_11:
	movabsq	$9223372036854775776, %rcx
	andq	%r14, %rcx
	je	.LBB140_14
	xorl	%edx, %edx
	.p2align	4
.LBB140_13:
	vmovups	(%r15,%rdx), %ymm0
	vmovups	%ymm0, (%rax,%rdx)
	addq	$32, %rdx
	cmpq	%rcx, %rdx
	jb	.LBB140_13
.LBB140_14:
	cmpq	%r14, %rcx
	je	.LBB140_16
	.p2align	4
.LBB140_15:
	movzbl	(%r15,%rcx), %edx
	movb	%dl, (%rax,%rcx)
	incq	%rcx
	cmpq	%rcx, %r14
	jne	.LBB140_15
.LBB140_16:
	movq	16(%rbx), %rax
	testq	%rax, %rax
	js	.LBB140_17
.LBB140_18:
	movq	%r12, 8(%rbx)
	jmp	.LBB140_19
.LBB140_10:
	movl	(%r15), %ecx
	movl	%ecx, (%rax)
	movl	-4(%r15,%r14), %ecx
	movl	%ecx, -4(%rax,%r14)
	movq	16(%rbx), %rax
	testq	%rax, %rax
	jns	.LBB140_18
.LBB140_17:
	movabsq	$-2233785415175766017, %rcx
	shlq	$56, %r12
	andq	%rcx, %rax
	orq	%r12, %rax
.LBB140_19:
	movabsq	$-2305843009213693953, %rcx
	andq	%rax, %rcx
	movq	%rcx, 16(%rbx)
.LBB140_20:
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.Lfunc_end7:
	.size	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])", .Lfunc_end7-"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"
	.cfi_endproc

	.prefalign	4, .Lfunc_end8, nop
	.type	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)",@function
"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$24, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	16(%rdi), %r15
	movabsq	$4611686018427387904, %r12
	movq	%rdi, %rbx
	testq	%r15, %r15
	js	.LBB141_11
	cmpq	%r12, %r15
	jae	.LBB141_3
	movq	8(%rbx), %rax
	cmpq	%rsi, %rax
	cmovgq	%rax, %rsi
	cmpq	$24, %rsi
	jl	.LBB141_5
	jmp	.LBB141_12
.LBB141_11:
	cmpq	$24, %rsi
	jge	.LBB141_12
	jmp	.LBB141_18
.LBB141_3:
	leaq	(,%r15,8), %rax
	cmpq	%rsi, %rax
	cmovgq	%rax, %rsi
	cmpq	$24, %rsi
	jge	.LBB141_12
.LBB141_5:
	movq	8(%rbx), %rdx
	movq	(%rbx), %r14
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, (%rsp)
	movq	%rdx, %rax
	orq	$-128, %rax
	shlq	$56, %rax
	movq	%rax, 16(%rsp)
	testq	%rdx, %rdx
	jle	.LBB141_7
	movq	%rsp, %rdi
	movq	%r14, %rsi
	callq	memcpy@PLT
.LBB141_7:
	cmpq	%r12, %r15
	jb	.LBB141_10
	lock		decq	-8(%r14)
	jne	.LBB141_10
	addq	$-8, %r14
	#MEMBARRIER
	movq	%r14, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB141_10:
	vmovups	8(%rsp), %xmm0
	movq	(%rsp), %rax
	movq	16(%rsp), %r15
	movq	%rax, (%rbx)
	vmovups	%xmm0, 8(%rbx)
	jmp	.LBB141_16
.LBB141_12:
	testq	%r12, %r15
	je	.LBB141_15
	movq	(%rbx), %rax
	movq	-8(%rax), %rax
	cmpq	$1, %rax
	jne	.LBB141_15
	leaq	(,%r15,8), %rax
	testq	%r15, %r15
	movl	$23, %ecx
	cmovnsq	%rax, %rcx
	cmpq	%rcx, %rsi
	jle	.LBB141_16
.LBB141_15:
	movq	%rbx, %rdi
	callq	"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])"@PLT
	movq	16(%rbx), %r15
.LBB141_16:
	testq	%r15, %r15
	js	.LBB141_18
	movq	(%rbx), %rbx
.LBB141_18:
	movq	%rbx, %rax
	addq	$24, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end8:
	.size	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)", .Lfunc_end8-"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"
	.cfi_endproc

	.prefalign	4, .Lfunc_end9, nop
	.type	"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])",@function
"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	16(%rdi), %r13
	movabsq	$4611686018427387904, %rax
	movq	%rdi, %rbx
	testq	%r13, %r13
	js	.LBB142_2
	movq	8(%rbx), %r12
	movq	(%rbx), %rbp
	leaq	(,%r13,8), %r15
	cmpq	%rax, %r13
	cmovbq	%r12, %r15
	addq	%r15, %r15
	jmp	.LBB142_3
.LBB142_2:
	movl	$1336, %eax
	movl	$46, %r15d
	movq	%rbx, %rbp
	bextrq	%rax, %r13, %r12
.LBB142_3:
	cmpq	%r15, %rsi
	cmovgq	%rsi, %r15
	addq	$7, %r15
	movq	%r15, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB142_22
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB142_24
	movq	%rax, %r14
	addq	$8, %r14
	movq	%r12, %rcx
	subq	$1, %rcx
	movq	$1, (%rax)
	jae	.LBB142_10
.LBB142_6:
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r13
	je	.LBB142_9
	movq	(%rbx), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB142_9
	addq	$-8, %rdi
	#MEMBARRIER
	vzeroupper
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movabsq	$4611686018427387904, %rax
.LBB142_9:
	sarq	$3, %r15
	movq	%r12, 8(%rbx)
	movq	%r14, (%rbx)
	orq	%rax, %r15
	movq	%r15, 16(%rbx)
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB142_10:
	.cfi_def_cfa_offset 80
	cmpq	$4, %r12
	jg	.LBB142_13
	movzbl	(%rbp), %edx
	movb	%dl, (%r14)
	movzbl	(%rbp,%rcx), %edx
	movb	%dl, (%r14,%rcx)
	cmpq	$3, %r12
	jl	.LBB142_6
	movzbl	1(%rbp), %ecx
	movb	%cl, 9(%rax)
	movzbl	-2(%rbp,%r12), %ecx
	movb	%cl, 6(%rax,%r12)
	jmp	.LBB142_6
.LBB142_13:
	cmpq	$16, %r12
	ja	.LBB142_16
	cmpq	$8, %r12
	jb	.LBB142_21
	movq	(%rbp), %rcx
	movq	%rcx, 8(%rax)
	movq	-8(%rbp,%r12), %rcx
	movq	%rcx, (%rax,%r12)
	jmp	.LBB142_6
.LBB142_16:
	movabsq	$9223372036854775776, %rax
	andq	%r12, %rax
	je	.LBB142_19
	xorl	%ecx, %ecx
	.p2align	4
.LBB142_18:
	vmovups	(%rbp,%rcx), %ymm0
	vmovups	%ymm0, (%r14,%rcx)
	addq	$32, %rcx
	cmpq	%rax, %rcx
	jb	.LBB142_18
.LBB142_19:
	cmpq	%r12, %rax
	je	.LBB142_6
	.p2align	4
.LBB142_20:
	movzbl	(%rbp,%rax), %ecx
	movb	%cl, (%r14,%rax)
	incq	%rax
	cmpq	%rax, %r12
	jne	.LBB142_20
	jmp	.LBB142_6
.LBB142_21:
	movl	(%rbp), %eax
	movl	%eax, (%r14)
	movl	-4(%rbp,%r12), %eax
	movl	%eax, -4(%r14,%r12)
	jmp	.LBB142_6
.LBB142_22:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	movl	$783, %edi
	movq	$35, 8(%rsp)
	movq	%rsp, %r8
	jmp	.LBB142_23
.LBB142_24:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	movq	$37, 8(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	movq	%rsp, %r8
	movl	$659, %edi
.LBB142_23:
	movq	%rax, (%rsp)
	movq	%rcx, 16(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end9:
	.size	"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])", .Lfunc_end9-"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])"
	.cfi_endproc

	.prefalign	4, .Lfunc_end10, nop
	.type	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false",@function
"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %rbx
	movq	%rsi, %r15
	movq	%rdi, %r14
	cmpq	$4097, %rdx
	jl	.LBB143_1
	movq	4096(%r14), %rdx
	movq	4104(%r14), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4096(%r14)
	movq	%r15, %rsi
	movq	%rbx, %rdx
	movq	4104(%r14), %rdi
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.LBB143_1:
	.cfi_def_cfa_offset 48
	movq	4096(%r14), %rdx
	leaq	(%rdx,%rbx), %rax
	cmpq	$4097, %rax
	jl	.LBB143_3
	movq	4104(%r14), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%r14)
.LBB143_3:
	testq	%rbx, %rbx
	je	.LBB143_15
	addq	%r14, %rdx
	cmpq	$4, %rbx
	jg	.LBB143_7
	movzbl	(%r15), %eax
	movb	%al, (%rdx)
	movzbl	-1(%r15,%rbx), %eax
	movb	%al, -1(%rdx,%rbx)
	cmpq	$3, %rbx
	jl	.LBB143_15
	movzbl	1(%r15), %eax
	movb	%al, 1(%rdx)
	movzbl	-2(%r15,%rbx), %eax
	movb	%al, -2(%rdx,%rbx)
	jmp	.LBB143_15
.LBB143_7:
	cmpq	$16, %rbx
	ja	.LBB143_11
	cmpq	$8, %rbx
	jb	.LBB143_10
	movq	(%r15), %rax
	movq	%rax, (%rdx)
	movq	-8(%r15,%rbx), %rax
	movq	%rax, -8(%rdx,%rbx)
	jmp	.LBB143_15
.LBB143_11:
	movabsq	$9223372036854775776, %r12
	andq	%rbx, %r12
	je	.LBB143_13
	movq	%rdx, %rdi
	movq	%rdx, %r13
	movq	%r15, %rsi
	movq	%r12, %rdx
	callq	memcpy@PLT
	movq	%r13, %rdx
.LBB143_13:
	cmpq	%rbx, %r12
	je	.LBB143_15
	movl	%ebx, %eax
	addq	%r12, %rdx
	addq	%r12, %r15
	andl	$31, %eax
	movq	%rdx, %rdi
	movq	%r15, %rsi
	movq	%rax, %rdx
	callq	memcpy@PLT
	jmp	.LBB143_15
.LBB143_10:
	movl	(%r15), %eax
	movl	%eax, (%rdx)
	movl	-4(%r15,%rbx), %eax
	movl	%eax, -4(%rdx,%rbx)
.LBB143_15:
	addq	%rbx, 4096(%r14)
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end10:
	.size	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false", .Lfunc_end10-"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"
	.cfi_endproc

	.prefalign	4, .Lfunc_end11, nop
	.type	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=true",@function
"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=true":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %rbx
	movq	%rsi, %r15
	movq	%rdi, %r14
	cmpq	$4097, %rdx
	jl	.LBB144_1
	movq	4096(%r14), %rdx
	movq	4104(%r14), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4096(%r14)
	movq	%r15, %rsi
	movq	%rbx, %rdx
	movq	4104(%r14), %rdi
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.LBB144_1:
	.cfi_def_cfa_offset 48
	movq	4096(%r14), %rdx
	leaq	(%rdx,%rbx), %rax
	cmpq	$4097, %rax
	jl	.LBB144_3
	movq	4104(%r14), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%r14)
.LBB144_3:
	testq	%rbx, %rbx
	je	.LBB144_15
	addq	%r14, %rdx
	cmpq	$4, %rbx
	jg	.LBB144_7
	movzbl	(%r15), %eax
	movb	%al, (%rdx)
	movzbl	-1(%r15,%rbx), %eax
	movb	%al, -1(%rdx,%rbx)
	cmpq	$3, %rbx
	jl	.LBB144_15
	movzbl	1(%r15), %eax
	movb	%al, 1(%rdx)
	movzbl	-2(%r15,%rbx), %eax
	movb	%al, -2(%rdx,%rbx)
	jmp	.LBB144_15
.LBB144_7:
	cmpq	$16, %rbx
	ja	.LBB144_11
	cmpq	$8, %rbx
	jb	.LBB144_10
	movq	(%r15), %rax
	movq	%rax, (%rdx)
	movq	-8(%r15,%rbx), %rax
	movq	%rax, -8(%rdx,%rbx)
	jmp	.LBB144_15
.LBB144_11:
	movabsq	$9223372036854775776, %r12
	andq	%rbx, %r12
	je	.LBB144_13
	movq	%rdx, %rdi
	movq	%rdx, %r13
	movq	%r15, %rsi
	movq	%r12, %rdx
	callq	memcpy@PLT
	movq	%r13, %rdx
.LBB144_13:
	cmpq	%rbx, %r12
	je	.LBB144_15
	movl	%ebx, %eax
	addq	%r12, %rdx
	addq	%r12, %r15
	andl	$31, %eax
	movq	%rdx, %rdi
	movq	%r15, %rsi
	movq	%rax, %rdx
	callq	memcpy@PLT
	jmp	.LBB144_15
.LBB144_10:
	movl	(%r15), %eax
	movl	%eax, (%rdx)
	movl	-4(%r15,%rbx), %eax
	movl	%eax, -4(%rdx,%rbx)
.LBB144_15:
	addq	%rbx, 4096(%r14)
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end11:
	.size	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=true", .Lfunc_end11-"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=true"
	.cfi_endproc

	.prefalign	4, .Lfunc_end12, nop
	.type	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false",@function
"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %rbx
	movq	%rsi, %r15
	movq	%rdi, %r14
	cmpq	$4097, %rdx
	jl	.LBB145_1
	movq	4104(%r14), %rax
	movq	4096(%r14), %rdx
	movq	%r14, %rsi
	movq	(%rax), %rdi
	callq	write@PLT
	movq	$0, 4096(%r14)
	movq	%r15, %rsi
	movq	%rbx, %rdx
	movq	4104(%r14), %rax
	movq	(%rax), %rdi
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	write@PLT
.LBB145_1:
	.cfi_def_cfa_offset 48
	movq	4096(%r14), %rdx
	leaq	(%rdx,%rbx), %rax
	cmpq	$4097, %rax
	jl	.LBB145_3
	movq	4104(%r14), %rax
	movq	%r14, %rsi
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%r14)
.LBB145_3:
	testq	%rbx, %rbx
	je	.LBB145_15
	addq	%r14, %rdx
	cmpq	$4, %rbx
	jg	.LBB145_7
	movzbl	(%r15), %eax
	movb	%al, (%rdx)
	movzbl	-1(%r15,%rbx), %eax
	movb	%al, -1(%rdx,%rbx)
	cmpq	$3, %rbx
	jl	.LBB145_15
	movzbl	1(%r15), %eax
	movb	%al, 1(%rdx)
	movzbl	-2(%r15,%rbx), %eax
	movb	%al, -2(%rdx,%rbx)
	jmp	.LBB145_15
.LBB145_7:
	cmpq	$16, %rbx
	ja	.LBB145_11
	cmpq	$8, %rbx
	jb	.LBB145_10
	movq	(%r15), %rax
	movq	%rax, (%rdx)
	movq	-8(%r15,%rbx), %rax
	movq	%rax, -8(%rdx,%rbx)
	jmp	.LBB145_15
.LBB145_11:
	movabsq	$9223372036854775776, %r12
	andq	%rbx, %r12
	je	.LBB145_13
	movq	%rdx, %rdi
	movq	%rdx, %r13
	movq	%r15, %rsi
	movq	%r12, %rdx
	callq	memcpy@PLT
	movq	%r13, %rdx
.LBB145_13:
	cmpq	%rbx, %r12
	je	.LBB145_15
	movl	%ebx, %eax
	addq	%r12, %rdx
	addq	%r12, %r15
	andl	$31, %eax
	movq	%rdx, %rdi
	movq	%r15, %rsi
	movq	%rax, %rdx
	callq	memcpy@PLT
	jmp	.LBB145_15
.LBB145_10:
	movl	(%r15), %eax
	movl	%eax, (%rdx)
	movl	-4(%r15,%rbx), %eax
	movl	%eax, -4(%rdx,%rbx)
.LBB145_15:
	addq	%rbx, 4096(%r14)
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end12:
	.size	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false", .Lfunc_end12-"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"
	.cfi_endproc

	.prefalign	4, .Lfunc_end13, nop
	.type	"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]",@function
"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]":
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	lock		decq	(%rdi)
	jne	.LBB146_2
	movq	%rdi, %rbx
	#MEMBARRIER
	movq	16(%rdi), %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	lock		decq	8(%rbx)
	jne	.LBB146_2
	#MEMBARRIER
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	KGEN_CompilerRT_AlignedFree@PLT
.LBB146_2:
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end13:
	.size	"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]", .Lfunc_end13-"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end14, nop
	.type	"std::reflection::location::SourceLocation::prefix[::Writable & ::AnyType](::SourceLocation,$0),T=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]",@function
"std::reflection::location::SourceLocation::prefix[::Writable & ::AnyType](::SourceLocation,$0),T=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r12
	movq	%rsi, %r13
	movabsq	$-9223372036854775808, %rax
	leaq	global_constant(%rip), %rsi
	movl	$3, %edx
	movq	%rdi, %rbp
	vxorps	%xmm0, %xmm0, %xmm0
	vmovups	%xmm0, (%r9)
	movq	%r9, %rdi
	movq	%r9, %rbx
	movq	%r8, %r14
	movq	%rcx, %r15
	movq	%rax, 16(%r9)
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	%rbp, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	movq	%r15, %rcx
	movq	%rbx, %r8
	callq	"std::reflection::location::SourceLocation::write_to[::Writer & ::AnyType](::SourceLocation,$0),writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	leaq	global_constant+4(%rip), %rsi
	movl	$2, %edx
	movq	%rbx, %rdi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	16(%r14), %rax
	testq	%rax, %rax
	js	.LBB147_1
	movq	8(%r14), %rdx
	movq	(%r14), %r14
	jmp	.LBB147_3
.LBB147_1:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB147_3:
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	leaq	global_constant+7(%rip), %rsi
	movq	%rbx, %rdi
	xorl	%edx, %edx
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.Lfunc_end14:
	.size	"std::reflection::location::SourceLocation::prefix[::Writable & ::AnyType](::SourceLocation,$0),T=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]", .Lfunc_end14-"std::reflection::location::SourceLocation::prefix[::Writable & ::AnyType](::SourceLocation,$0),T=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end15, nop
	.type	"std::reflection::location::SourceLocation::write_to[::Writer & ::AnyType](::SourceLocation,$0),writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]",@function
"std::reflection::location::SourceLocation::write_to[::Writer & ::AnyType](::SourceLocation,$0),writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4168, %rsp
	.cfi_def_cfa_offset 4224
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	16(%r8), %rbp
	movq	%rcx, %r12
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	movq	$1, 40(%rsp)
	movq	$1, 16(%rsp)
	movq	%r8, %r15
	movq	%rdx, %r13
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movq	%rax, 32(%rsp)
	movq	%rax, 8(%rsp)
	movq	%rcx, 48(%rsp)
	movq	%rcx, 24(%rsp)
	testq	%rbp, %rbp
	js	.LBB148_1
	movq	8(%r15), %rax
	jmp	.LBB148_3
.LBB148_1:
	movl	$1336, %eax
	bextrq	%rax, %rbp, %rax
.LBB148_3:
	leaq	1(%r12,%rax), %rsi
	movq	%r14, %rdi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	leaq	1(%rax), %rsi
	movq	%rbx, %rdi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	cmpq	$23, %rax
	jg	.LBB148_8
	movq	%r15, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rsi
	movl	$1, %edx
	movq	%r15, %rdi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movq	24(%rsp), %rax
	testq	%rax, %rax
	js	.LBB148_5
	movq	8(%rsp), %rsi
	movq	16(%rsp), %rdx
	jmp	.LBB148_7
.LBB148_8:
	testq	%rbp, %rbp
	js	.LBB148_13
	movabsq	$4611686018427387904, %rcx
	cmpq	%rcx, %rbp
	jae	.LBB148_11
	movq	8(%r15), %rbp
	cmpq	%rbp, %rax
	jg	.LBB148_13
	jmp	.LBB148_14
.LBB148_5:
	movl	$1336, %ecx
	leaq	8(%rsp), %rsi
	bextrq	%rcx, %rax, %rdx
.LBB148_7:
	movq	%r15, %rdi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	%rbx, %rdi
	movq	%r15, %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movabsq	$4611686018427387904, %rbx
	testq	%rbx, 24(%rsp)
	jne	.LBB148_32
	jmp	.LBB148_34
.LBB148_11:
	shlq	$3, %rbp
	cmpq	%rbp, %rax
	jle	.LBB148_14
.LBB148_13:
	movq	%r15, %rdi
	movq	%rax, %rsi
	callq	"std::collections::string::string::String::_realloc_mutable(::String,::SIMD[DType.int, 1])"@PLT
.LBB148_14:
	leaq	56(%rsp), %rdi
	movq	$0, 4152(%rsp)
	movq	%r15, 4160(%rsp)
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4152(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB148_16
	movq	4160(%rsp), %rdi
	leaq	56(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4152(%rsp)
.LBB148_16:
	movq	48(%rsp), %rax
	testq	%rax, %rax
	js	.LBB148_17
	movq	32(%rsp), %rsi
	movq	40(%rsp), %rdx
	jmp	.LBB148_19
.LBB148_17:
	movl	$1336, %ecx
	leaq	32(%rsp), %rsi
	bextrq	%rcx, %rax, %rdx
.LBB148_19:
	leaq	56(%rsp), %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4152(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB148_21
	movq	4160(%rsp), %rdi
	leaq	56(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4152(%rsp)
.LBB148_21:
	leaq	56(%rsp), %rsi
	movq	%r14, %rdi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4152(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB148_23
	movq	4160(%rsp), %rdi
	leaq	56(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4152(%rsp)
.LBB148_23:
	movq	24(%rsp), %rax
	testq	%rax, %rax
	js	.LBB148_24
	movq	8(%rsp), %rsi
	movq	16(%rsp), %rdx
	jmp	.LBB148_26
.LBB148_24:
	movl	$1336, %ecx
	leaq	8(%rsp), %rsi
	bextrq	%rcx, %rax, %rdx
.LBB148_26:
	leaq	56(%rsp), %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4152(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB148_28
	movq	4160(%rsp), %rdi
	leaq	56(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4152(%rsp)
.LBB148_28:
	leaq	56(%rsp), %r14
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4152(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB148_30
	movq	4160(%rsp), %rdi
	leaq	56(%rsp), %r14
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4152(%rsp)
.LBB148_30:
	movq	4160(%rsp), %rdi
	movq	%r14, %rsi
	movabsq	$4611686018427387904, %rbx
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	testq	%rbx, 24(%rsp)
	je	.LBB148_34
.LBB148_32:
	movq	8(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB148_34
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB148_34:
	testq	%rbx, 48(%rsp)
	je	.LBB148_37
	movq	32(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB148_37
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB148_37:
	addq	$4168, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end15:
	.size	"std::reflection::location::SourceLocation::write_to[::Writer & ::AnyType](::SourceLocation,$0),writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]", .Lfunc_end15-"std::reflection::location::SourceLocation::write_to[::Writer & ::AnyType](::SourceLocation,$0),writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end16, nop
	.type	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]",@function
"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$72, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	testq	%rdi, %rdi
	js	.LBB149_1
	leaq	static_string_2d06800538d394c2(%rip), %rsi
	movq	%rbx, %rdi
	xorl	%edx, %edx
	callq	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"@PLT
	testq	%r14, %r14
	je	.LBB149_16
	leaq	static_string_978d8d34847e5196(%rip), %rcx
	movabsq	$-3689348814741910323, %rsi
	xorl	%eax, %eax
	movb	$0, 71(%rsp)
	.p2align	4
.LBB149_12:
	movq	%r14, %rdx
	mulxq	%rsi, %rdx, %rdx
	movq	%r14, %r8
	shrq	$3, %rdx
	leaq	(%rdx,%rdx), %rdi
	leaq	(%rdi,%rdi,4), %rdi
	subq	%rdi, %r8
	movzbl	(%rcx,%r8), %edi
	movb	%dil, 70(%rsp,%rax)
	decq	%rax
	cmpq	$9, %r14
	movq	%rdx, %r14
	ja	.LBB149_12
	leaq	71(%rsp,%rax), %rsi
	negq	%rax
	movq	%rbx, %rdi
	movq	%rax, %rdx
	jmp	.LBB149_14
.LBB149_1:
	movq	16(%rbx), %rax
	testq	%rax, %rax
	js	.LBB149_2
	movq	8(%rbx), %r12
	jmp	.LBB149_4
.LBB149_16:
	leaq	7(%rsp), %rsi
	movl	$1, %edx
	movq	%rbx, %rdi
	movw	$48, 7(%rsp)
	callq	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=true,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"@PLT
	jmp	.LBB149_15
.LBB149_2:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %r12
.LBB149_4:
	leaq	1(%r12), %r15
	movq	%rbx, %rdi
	movq	%r15, %rsi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movb	$45, (%rax,%r12)
	movq	16(%rbx), %rax
	testq	%rax, %rax
	js	.LBB149_5
	movq	%r15, 8(%rbx)
	jmp	.LBB149_7
.LBB149_5:
	movabsq	$-2233785415175766017, %rcx
	shlq	$56, %r15
	andq	%rcx, %rax
	orq	%r15, %rax
.LBB149_7:
	movabsq	$-2305843009213693953, %rcx
	leaq	static_string_2d06800538d394c2(%rip), %rsi
	xorl	%r15d, %r15d
	movq	%rbx, %rdi
	xorl	%edx, %edx
	andq	%rax, %rcx
	movq	%rcx, 16(%rbx)
	callq	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"@PLT
	movabsq	$7378697629483820647, %rcx
	leaq	static_string_978d8d34847e5196(%rip), %rsi
	movabsq	$-7378697629483820647, %rdi
	movb	$0, 71(%rsp)
	.p2align	4
.LBB149_8:
	movq	%r14, %rax
	imulq	%rcx
	movq	%r14, %r9
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	addq	%rdx, %rdx
	leaq	(%rdx,%rdx,4), %rax
	movq	%r14, %rdx
	subq	%rax, %rdx
	testq	%r14, %r14
	movq	$-10, %rax
	setns	%r8b
	testq	%rdx, %rdx
	cmoveq	%rdx, %rax
	sarq	$63, %r9
	andnq	%rax, %r9, %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	negq	%rdx
	cmovsq	%rax, %rdx
	movzbl	(%rdx,%rsi), %eax
	movb	%al, 70(%rsp,%r15)
	movq	%r14, %rax
	imulq	%rdi
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	leaq	(%rax,%rax,4), %rax
	addq	%r14, %rax
	setne	%al
	decq	%r15
	andb	%r8b, %al
	movzbl	%al, %r14d
	subq	%rdx, %r14
	jne	.LBB149_8
	leaq	71(%rsp,%r15), %rsi
	negq	%r15
	movq	%rbx, %rdi
	movq	%r15, %rdx
.LBB149_14:
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.LBB149_15:
	addq	$72, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end16:
	.size	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]", .Lfunc_end16-"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end17, nop
	.type	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]",@function
"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]":
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$72, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	testq	%rdi, %rdi
	js	.LBB150_7
	movq	4096(%rbx), %rdx
	je	.LBB150_15
	cmpq	$4096, %rdx
	jle	.LBB150_4
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4096(%rbx)
.LBB150_4:
	leaq	static_string_978d8d34847e5196(%rip), %rcx
	movabsq	$-3689348814741910323, %rsi
	xorl	%eax, %eax
	movb	$0, 71(%rsp)
	.p2align	4
.LBB150_5:
	movq	%r14, %rdx
	mulxq	%rsi, %rdx, %rdx
	movq	%r14, %r8
	shrq	$3, %rdx
	leaq	(%rdx,%rdx), %rdi
	leaq	(%rdi,%rdi,4), %rdi
	subq	%rdi, %r8
	movzbl	(%rcx,%r8), %edi
	movb	%dil, 70(%rsp,%rax)
	decq	%rax
	cmpq	$9, %r14
	movq	%rdx, %r14
	ja	.LBB150_5
	leaq	71(%rsp,%rax), %rsi
	negq	%rax
	movq	%rbx, %rdi
	movq	%rax, %rdx
	jmp	.LBB150_14
.LBB150_7:
	movq	4096(%rbx), %rdx
	leaq	1(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB150_9
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rbx)
.LBB150_9:
	movb	$45, (%rbx,%rdx)
	movq	4096(%rbx), %rdx
	incq	%rdx
	movq	%rdx, 4096(%rbx)
	cmpq	$4097, %rdx
	jl	.LBB150_11
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4096(%rbx)
.LBB150_11:
	movabsq	$7378697629483820647, %rsi
	leaq	static_string_978d8d34847e5196(%rip), %rdi
	movabsq	$-7378697629483820647, %r8
	xorl	%ecx, %ecx
	movb	$0, 71(%rsp)
	.p2align	4
.LBB150_12:
	movq	%r14, %rax
	imulq	%rsi
	movq	%r14, %r10
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	addq	%rdx, %rdx
	leaq	(%rdx,%rdx,4), %rax
	movq	%r14, %rdx
	subq	%rax, %rdx
	testq	%r14, %r14
	movq	$-10, %rax
	setns	%r9b
	testq	%rdx, %rdx
	cmoveq	%rdx, %rax
	sarq	$63, %r10
	andnq	%rax, %r10, %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	negq	%rdx
	cmovsq	%rax, %rdx
	movzbl	(%rdx,%rdi), %eax
	movb	%al, 70(%rsp,%rcx)
	movq	%r14, %rax
	imulq	%r8
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	leaq	(%rax,%rax,4), %rax
	addq	%r14, %rax
	setne	%al
	decq	%rcx
	andb	%r9b, %al
	movzbl	%al, %r14d
	subq	%rdx, %r14
	jne	.LBB150_12
	leaq	71(%rsp,%rcx), %rsi
	negq	%rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
.LBB150_14:
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=true"@PLT
	addq	$72, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB150_15:
	.cfi_def_cfa_offset 96
	cmpq	$4096, %rdx
	jl	.LBB150_17
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rbx)
.LBB150_17:
	movb	$48, (%rbx,%rdx)
	incq	4096(%rbx)
	addq	$72, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end17:
	.size	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]", .Lfunc_end17-"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end18, nop
	.type	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]",@function
"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r15
	testq	%rdi, %rdi
	js	.LBB151_1
	movq	4096(%rbx), %r14
	cmpq	$4096, %r14
	jle	.LBB151_8
	movq	4104(%rbx), %rax
	movq	%rbx, %rsi
	movq	%r14, %rdx
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%r14d, %r14d
	movq	$0, 4096(%rbx)
	testq	%r15, %r15
	jne	.LBB151_13
	jmp	.LBB151_11
.LBB151_1:
	movq	4096(%rbx), %rdx
	leaq	1(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB151_3
	movq	4104(%rbx), %rax
	movq	%rbx, %rsi
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rbx)
.LBB151_3:
	movb	$45, (%rbx,%rdx)
	movq	4096(%rbx), %r14
	incq	%r14
	movq	%r14, 4096(%rbx)
	cmpq	$4097, %r14
	jl	.LBB151_5
	movq	4104(%rbx), %rax
	movq	%rbx, %rsi
	movq	%r14, %rdx
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%r14d, %r14d
	movq	$0, 4096(%rbx)
.LBB151_5:
	movl	$63, %r13d
	movabsq	$7378697629483820647, %rcx
	leaq	static_string_978d8d34847e5196(%rip), %rsi
	movabsq	$-7378697629483820647, %rdi
	movb	$0, 71(%rsp)
	.p2align	4
.LBB151_6:
	movq	%r15, %rax
	imulq	%rcx
	movq	%r15, %r9
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	addq	%rdx, %rdx
	leaq	(%rdx,%rdx,4), %rax
	movq	%r15, %rdx
	subq	%rax, %rdx
	testq	%r15, %r15
	movq	$-10, %rax
	setns	%r8b
	testq	%rdx, %rdx
	cmoveq	%rdx, %rax
	sarq	$63, %r9
	andnq	%rax, %r9, %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	negq	%rdx
	cmovsq	%rax, %rdx
	movq	%r15, %rax
	movzbl	(%rdx,%rsi), %ebp
	imulq	%rdi
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	leaq	(%rax,%rax,4), %rax
	movb	%bpl, 7(%rsp,%r13)
	decq	%r13
	addq	%r15, %rax
	setne	%al
	andb	%r8b, %al
	movzbl	%al, %r15d
	subq	%rdx, %r15
	jne	.LBB151_6
	jmp	.LBB151_15
.LBB151_8:
	testq	%r15, %r15
	je	.LBB151_9
.LBB151_13:
	movl	$63, %r13d
	leaq	static_string_978d8d34847e5196(%rip), %rax
	movabsq	$-3689348814741910323, %rcx
	movb	$0, 71(%rsp)
	.p2align	4
.LBB151_14:
	movq	%r15, %rdx
	mulxq	%rcx, %rdx, %rdx
	movq	%r15, %rdi
	shrq	$3, %rdx
	leaq	(%rdx,%rdx), %rsi
	leaq	(%rsi,%rsi,4), %rsi
	subq	%rsi, %rdi
	movzbl	(%rax,%rdi), %ebp
	movb	%bpl, 7(%rsp,%r13)
	decq	%r13
	cmpq	$10, %r15
	movq	%rdx, %r15
	jae	.LBB151_14
.LBB151_15:
	movl	$63, %r15d
	leaq	8(%rsp,%r13), %r12
	subq	%r13, %r15
	cmpq	$4097, %r15
	jl	.LBB151_17
	movq	4104(%rbx), %rax
	movq	%rbx, %rsi
	movq	%r14, %rdx
	movq	(%rax), %rdi
	callq	write@PLT
	movq	$0, 4096(%rbx)
	movq	%r12, %rsi
	movq	%r15, %rdx
	movq	4104(%rbx), %rax
	movq	(%rax), %rdi
	callq	write@PLT
	jmp	.LBB151_32
.LBB151_17:
	leaq	(%r14,%r15), %rax
	cmpq	$4097, %rax
	jl	.LBB151_19
	movq	4104(%rbx), %rax
	movq	%rbx, %rsi
	movq	%r14, %rdx
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%r14d, %r14d
	movq	$0, 4096(%rbx)
.LBB151_19:
	cmpq	$63, %r13
	jne	.LBB151_20
.LBB151_31:
	addq	%r15, 4096(%rbx)
	jmp	.LBB151_32
.LBB151_20:
	addq	%rbx, %r14
	cmpq	$4, %r15
	jg	.LBB151_23
	movzbl	70(%rsp), %ecx
	movl	$62, %eax
	movb	%bpl, (%r14)
	subq	%r13, %rax
	movb	%cl, (%r14,%rax)
	cmpq	$3, %r15
	jl	.LBB151_31
	leaq	7(%rsp,%r13), %rcx
	movzbl	69(%rsp), %edx
	movl	$61, %eax
	subq	%r13, %rax
	movzbl	2(%rcx), %ecx
	movb	%cl, 1(%r14)
	movb	%dl, (%r14,%rax)
	jmp	.LBB151_31
.LBB151_9:
	cmpq	$4096, %r14
	jne	.LBB151_11
	movq	4104(%rbx), %rax
	movq	%rbx, %rsi
	movq	%r14, %rdx
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%r14d, %r14d
	movq	$0, 4096(%rbx)
.LBB151_11:
	movb	$48, (%rbx,%r14)
	incq	4096(%rbx)
.LBB151_32:
	addq	$72, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB151_23:
	.cfi_def_cfa_offset 128
	cmpq	$16, %r15
	ja	.LBB151_27
	cmpq	$8, %r15
	jb	.LBB151_26
	movq	(%r12), %rcx
	movq	63(%rsp), %rdx
	movl	$55, %eax
	subq	%r13, %rax
	movq	%rcx, (%r14)
	movq	%rdx, (%r14,%rax)
	jmp	.LBB151_31
.LBB151_27:
	movabsq	$9223372036854775776, %r13
	andq	%r15, %r13
	je	.LBB151_29
	movq	%r14, %rdi
	movq	%r12, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
.LBB151_29:
	cmpq	%r15, %r13
	je	.LBB151_31
	movl	%r15d, %edx
	addq	%r13, %r14
	addq	%r13, %r12
	andl	$31, %edx
	movq	%r14, %rdi
	movq	%r12, %rsi
	callq	memcpy@PLT
	jmp	.LBB151_31
.LBB151_26:
	movl	(%r12), %ecx
	movl	67(%rsp), %edx
	movl	$59, %eax
	subq	%r13, %rax
	movl	%ecx, (%r14)
	movl	%edx, (%r14,%rax)
	jmp	.LBB151_31
.Lfunc_end18:
	.size	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]", .Lfunc_end18-"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end19, nop
	.type	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]",@function
"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]":
	testq	%rdi, %rdi
	je	.LBB152_6
	movq	%rdi, %rax
	shrq	$63, %rax
	leaq	-1(%rsi,%rax), %rsi
	testq	%rdi, %rdi
	js	.LBB152_2
	movabsq	$-3689348814741910323, %rax
	.p2align	4
.LBB152_5:
	movq	%rdi, %rdx
	mulxq	%rax, %rcx, %rcx
	incq	%rsi
	shrq	$3, %rcx
	cmpq	$10, %rdi
	movq	%rcx, %rdi
	jae	.LBB152_5
	jmp	.LBB152_6
.LBB152_2:
	movabsq	$-7378697629483820647, %rcx
	.p2align	4
.LBB152_3:
	movq	%rdi, %rax
	imulq	%rcx
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	leaq	(%rax,%rax,4), %rax
	addq	%rdi, %rax
	setne	%al
	testq	%rdi, %rdi
	setns	%dil
	incq	%rsi
	andb	%al, %dil
	movzbl	%dil, %edi
	subq	%rdx, %rdi
	jne	.LBB152_3
.LBB152_6:
	incq	%rsi
	movq	%rsi, %rax
	retq
.Lfunc_end19:
	.size	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]", .Lfunc_end19-"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"

	.prefalign	4, .Lfunc_end20, nop
	.type	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]",@function
"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$72, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	testq	%rdi, %rdi
	js	.LBB153_1
	leaq	static_string_2d06800538d394c2(%rip), %rsi
	movq	%rbx, %rdi
	xorl	%edx, %edx
	callq	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"@PLT
	testq	%r14, %r14
	je	.LBB153_16
	leaq	static_string_978d8d34847e5196(%rip), %rcx
	movabsq	$-3689348814741910323, %rsi
	xorl	%eax, %eax
	movb	$0, 71(%rsp)
	.p2align	4
.LBB153_12:
	movq	%r14, %rdx
	mulxq	%rsi, %rdx, %rdx
	movq	%r14, %r8
	shrq	$3, %rdx
	leaq	(%rdx,%rdx), %rdi
	leaq	(%rdi,%rdi,4), %rdi
	subq	%rdi, %r8
	movzbl	(%rcx,%r8), %edi
	movb	%dil, 70(%rsp,%rax)
	decq	%rax
	cmpq	$9, %r14
	movq	%rdx, %r14
	ja	.LBB153_12
	leaq	71(%rsp,%rax), %rsi
	negq	%rax
	movq	%rbx, %rdi
	movq	%rax, %rdx
	jmp	.LBB153_14
.LBB153_1:
	movq	16(%rbx), %rax
	testq	%rax, %rax
	js	.LBB153_2
	movq	8(%rbx), %r12
	jmp	.LBB153_4
.LBB153_16:
	leaq	7(%rsp), %rsi
	movl	$1, %edx
	movq	%rbx, %rdi
	movw	$48, 7(%rsp)
	callq	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=true,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"@PLT
	jmp	.LBB153_15
.LBB153_2:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %r12
.LBB153_4:
	leaq	1(%r12), %r15
	movq	%rbx, %rdi
	movq	%r15, %rsi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movb	$45, (%rax,%r12)
	movq	16(%rbx), %rax
	testq	%rax, %rax
	js	.LBB153_5
	movq	%r15, 8(%rbx)
	jmp	.LBB153_7
.LBB153_5:
	movabsq	$-2233785415175766017, %rcx
	shlq	$56, %r15
	andq	%rcx, %rax
	orq	%r15, %rax
.LBB153_7:
	movabsq	$-2305843009213693953, %rcx
	leaq	static_string_2d06800538d394c2(%rip), %rsi
	xorl	%r15d, %r15d
	movq	%rbx, %rdi
	xorl	%edx, %edx
	andq	%rax, %rcx
	movq	%rcx, 16(%rbx)
	callq	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"@PLT
	movabsq	$7378697629483820647, %rcx
	leaq	static_string_978d8d34847e5196(%rip), %rsi
	movabsq	$-7378697629483820647, %rdi
	movb	$0, 71(%rsp)
	.p2align	4
.LBB153_8:
	movq	%r14, %rax
	imulq	%rcx
	movq	%r14, %r9
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	addq	%rdx, %rdx
	leaq	(%rdx,%rdx,4), %rax
	movq	%r14, %rdx
	subq	%rax, %rdx
	testq	%r14, %r14
	movq	$-10, %rax
	setns	%r8b
	testq	%rdx, %rdx
	cmoveq	%rdx, %rax
	sarq	$63, %r9
	andnq	%rax, %r9, %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	negq	%rdx
	cmovsq	%rax, %rdx
	movzbl	(%rdx,%rsi), %eax
	movb	%al, 70(%rsp,%r15)
	movq	%r14, %rax
	imulq	%rdi
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	leaq	(%rax,%rax,4), %rax
	addq	%r14, %rax
	setne	%al
	decq	%r15
	andb	%r8b, %al
	movzbl	%al, %r14d
	subq	%rdx, %r14
	jne	.LBB153_8
	leaq	71(%rsp,%r15), %rsi
	negq	%r15
	movq	%rbx, %rdi
	movq	%r15, %rdx
.LBB153_14:
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.LBB153_15:
	addq	$72, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end20:
	.size	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]", .Lfunc_end20-"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end21, nop
	.type	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]",@function
"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]":
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$72, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	testq	%rdi, %rdi
	js	.LBB154_7
	movq	4096(%rbx), %rdx
	je	.LBB154_15
	cmpq	$4096, %rdx
	jle	.LBB154_4
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4096(%rbx)
.LBB154_4:
	leaq	static_string_978d8d34847e5196(%rip), %rcx
	movabsq	$-3689348814741910323, %rsi
	xorl	%eax, %eax
	movb	$0, 71(%rsp)
	.p2align	4
.LBB154_5:
	movq	%r14, %rdx
	mulxq	%rsi, %rdx, %rdx
	movq	%r14, %r8
	shrq	$3, %rdx
	leaq	(%rdx,%rdx), %rdi
	leaq	(%rdi,%rdi,4), %rdi
	subq	%rdi, %r8
	movzbl	(%rcx,%r8), %edi
	movb	%dil, 70(%rsp,%rax)
	decq	%rax
	cmpq	$9, %r14
	movq	%rdx, %r14
	ja	.LBB154_5
	leaq	71(%rsp,%rax), %rsi
	negq	%rax
	movq	%rbx, %rdi
	movq	%rax, %rdx
	jmp	.LBB154_14
.LBB154_7:
	movq	4096(%rbx), %rdx
	leaq	1(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB154_9
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rbx)
.LBB154_9:
	movb	$45, (%rbx,%rdx)
	movq	4096(%rbx), %rdx
	incq	%rdx
	movq	%rdx, 4096(%rbx)
	cmpq	$4097, %rdx
	jl	.LBB154_11
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	$0, 4096(%rbx)
.LBB154_11:
	movabsq	$7378697629483820647, %rsi
	leaq	static_string_978d8d34847e5196(%rip), %rdi
	movabsq	$-7378697629483820647, %r8
	xorl	%ecx, %ecx
	movb	$0, 71(%rsp)
	.p2align	4
.LBB154_12:
	movq	%r14, %rax
	imulq	%rsi
	movq	%r14, %r10
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	addq	%rdx, %rdx
	leaq	(%rdx,%rdx,4), %rax
	movq	%r14, %rdx
	subq	%rax, %rdx
	testq	%r14, %r14
	movq	$-10, %rax
	setns	%r9b
	testq	%rdx, %rdx
	cmoveq	%rdx, %rax
	sarq	$63, %r10
	andnq	%rax, %r10, %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	negq	%rdx
	cmovsq	%rax, %rdx
	movzbl	(%rdx,%rdi), %eax
	movb	%al, 70(%rsp,%rcx)
	movq	%r14, %rax
	imulq	%r8
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	leaq	(%rax,%rax,4), %rax
	addq	%r14, %rax
	setne	%al
	decq	%rcx
	andb	%r9b, %al
	movzbl	%al, %r14d
	subq	%rdx, %r14
	jne	.LBB154_12
	leaq	71(%rsp,%rcx), %rsi
	negq	%rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
.LBB154_14:
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=true"@PLT
	addq	$72, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB154_15:
	.cfi_def_cfa_offset 96
	cmpq	$4096, %rdx
	jl	.LBB154_17
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rbx)
.LBB154_17:
	movb	$48, (%rbx,%rdx)
	incq	4096(%rbx)
	addq	$72, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end21:
	.size	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]", .Lfunc_end21-"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end22, nop
	.type	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]",@function
"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]":
	testq	%rdi, %rdi
	je	.LBB155_6
	movq	%rdi, %rax
	shrq	$63, %rax
	leaq	-1(%rsi,%rax), %rsi
	testq	%rdi, %rdi
	js	.LBB155_2
	movabsq	$-3689348814741910323, %rax
	.p2align	4
.LBB155_5:
	movq	%rdi, %rdx
	mulxq	%rax, %rcx, %rcx
	incq	%rsi
	shrq	$3, %rcx
	cmpq	$10, %rdi
	movq	%rcx, %rdi
	jae	.LBB155_5
	jmp	.LBB155_6
.LBB155_2:
	movabsq	$-7378697629483820647, %rcx
	.p2align	4
.LBB155_3:
	movq	%rdi, %rax
	imulq	%rcx
	movq	%rdx, %rax
	shrq	$63, %rax
	sarq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	leaq	(%rax,%rax,4), %rax
	addq	%rdi, %rax
	setne	%al
	testq	%rdi, %rdi
	setns	%dil
	incq	%rsi
	andb	%al, %dil
	movzbl	%dil, %edi
	subq	%rdx, %rdi
	jne	.LBB155_3
.LBB155_6:
	incq	%rsi
	movq	%rsi, %rax
	retq
.Lfunc_end22:
	.size	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]", .Lfunc_end22-"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=si64,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"

	.prefalign	4, .Lfunc_end23, nop
	.type	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]",@function
"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]":
	.cfi_startproc
	leaq	static_string_ba261bf194cae289(%rip), %rsi
	movl	$4, %edx
	jmp	"std::collections::string::string::String::write[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](::String,*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>]]"@PLT
.Lfunc_end23:
	.size	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]", .Lfunc_end23-"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end24, nop
	.type	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]",@function
"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]":
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	4096(%rdi), %rdx
	movq	%rdi, %rbx
	leaq	4(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB157_2
	movq	4104(%rbx), %rdi
	movq	%rbx, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4096(%rbx)
.LBB157_2:
	movl	$1701736302, (%rbx,%rdx)
	addq	$4, 4096(%rbx)
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end24:
	.size	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]", .Lfunc_end24-"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end25, nop
	.type	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]",@function
"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]":
	leaq	4(%rdi), %rax
	retq
.Lfunc_end25:
	.size	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]", .Lfunc_end25-"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"

	.prefalign	4, .Lfunc_end26, nop
	.type	"std::io::io::_flush(::FileDescriptor)",@function
"std::io::io::_flush(::FileDescriptor)":
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	callq	dup@PLT
	leaq	static_string_0d78baac08237ddb(%rip), %rsi
	movl	%eax, %edi
	callq	fdopen@PLT
	movq	%rax, %rdi
	movq	%rax, %rbx
	callq	fflush@PLT
	movq	%rbx, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	fclose@PLT
.Lfunc_end26:
	.size	"std::io::io::_flush(::FileDescriptor)", .Lfunc_end26-"std::io::io::_flush(::FileDescriptor)"
	.cfi_endproc

	.prefalign	4, .Lfunc_end27, nop
	.type	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]",@function
"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%r8, %r11
	movq	%rcx, %r10
	movq	%rdx, %rax
	movq	%rsi, %r9
	movq	%rdi, %r8
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_bbe01a6a523daf15(%rip), %rbx
	leaq	static_string_c44bdff4074eecdb(%rip), %r14
	leaq	static_string_7f1562353e292282(%rip), %r15
	leaq	static_string_31203c1a2bdb78cc(%rip), %rdi
	leaq	static_string_a8d4ace0dc8d360e(%rip), %rdx
	movl	$6, %esi
	movl	$1, %ecx
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	%rbx
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	$2
	.cfi_adjust_cfa_offset 8
	pushq	%r15
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	"std::io::io::print[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2],::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5],::Bool,LITOrigin[$7._mlir_value],::Origin[$7, $8]](*$0,sep:::StringSpan[$4, $5, $6],end:::StringSpan[$7, $8, $9],flush:::Bool,file:::FileDescriptor$)_REMOVED_ARG,Ts.values`=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$96, %rsp
	.cfi_adjust_cfa_offset -96
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end27:
	.size	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]", .Lfunc_end27-"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"
	.cfi_endproc

	.prefalign	4, .Lfunc_end28, nop
	.type	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }",@function
"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }":
	xorl	%eax, %eax
	retq
.Lfunc_end28:
	.size	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }", .Lfunc_end28-"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"

	.prefalign	4, .Lfunc_end29, nop
	.type	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])",@function
"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$4144, %rsp
	.cfi_def_cfa_offset 4176
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movabsq	$-9223372036854775808, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, (%rsp)
	movq	%rdi, %rbx
	movq	%rax, 16(%rsp)
	testq	%rdi, %rdi
	je	.LBB162_14
	leaq	-1(%rax), %rcx
	xorl	%r14d, %r14d
	.p2align	4
.LBB162_2:
	cmpb	$0, (%rbx,%r14)
	je	.LBB162_6
	incq	%r14
	cmpq	%r14, %rcx
	jne	.LBB162_2
.LBB162_4:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	32(%rsp), %r8
	movl	$783, %edi
	movq	$35, 40(%rsp)
	jmp	.LBB162_5
.LBB162_6:
	cmpq	$23, %r14
	ja	.LBB162_8
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, (%rsp)
	movq	%rax, 16(%rsp)
	movq	%rsp, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	jmp	.LBB162_13
.LBB162_8:
	movq	%r14, %r15
	addq	$7, %r15
	movq	%r15, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB162_4
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB162_15
	sarq	$3, %r15
	movabsq	$4611686018427387904, %rcx
	movq	$1, (%rax)
	addq	$8, %rax
	movq	$0, 4128(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdx
	orq	%r15, %rcx
	leaq	32(%rsp), %r15
	movq	%rax, (%rsp)
	movq	$0, 8(%rsp)
	movq	%rsp, %rax
	movq	%rax, 4136(%rsp)
	movq	%r15, %rdi
	movq	%rcx, 16(%rsp)
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4128(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB162_12
	movq	4136(%rsp), %rdi
	leaq	32(%rsp), %r15
	movq	%r15, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4128(%rsp)
.LBB162_12:
	movq	4136(%rsp), %rdi
	movq	%r15, %rsi
.LBB162_13:
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
.LBB162_14:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_strfree@PLT
	movq	(%rsp), %rax
	movq	8(%rsp), %rdx
	movq	16(%rsp), %rcx
	addq	$4144, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB162_15:
	.cfi_def_cfa_offset 4176
	leaq	static_string_09e773a88105e290(%rip), %rax
	movabsq	$2305843009213693952, %rcx
	movq	$37, 40(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	32(%rsp), %r8
	movl	$659, %edi
.LBB162_5:
	movq	%rax, 32(%rsp)
	movq	%rcx, 48(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end29:
	.size	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])", .Lfunc_end29-"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"
	.cfi_endproc

	.prefalign	4, .Lfunc_end30, nop
	.type	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG",@function
"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$104, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	movq	%rsi, %rdi
	movq	%r9, %r14
	movq	%r8, %r15
	movq	%rcx, %r12
	movq	%rdx, %rbp
	movabsq	$2305843009213693952, %r13
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	movq	%rax, 8(%rsp)
	movq	%rdx, 16(%rsp)
	movq	%rcx, 24(%rsp)
	movq	16(%rbp), %rax
	testq	%rax, %rax
	js	.LBB163_3
	movq	8(%rbp), %rcx
	testq	%rcx, %rcx
	jle	.LBB163_2
	movq	(%rbp), %rbp
	jmp	.LBB163_6
.LBB163_3:
	movabsq	$2233785415175766016, %rcx
	testq	%rcx, %rax
	je	.LBB163_2
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rcx
.LBB163_6:
	leaq	static_string_a8d4ace0dc8d360e(%rip), %rdi
	movl	$1, %esi
	movq	%rbp, %rdx
	callq	"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[DType.uint8, 1], $2, AddressSpace.GENERIC],::Span[$3, $4, ::SIMD[DType.uint8, 1], $5, AddressSpace.GENERIC]),lhs.mut`2x=false,rhs.mut`2x3=false"@PLT
	movq	%rcx, %r13
	jmp	.LBB163_7
.LBB163_2:
	xorl	%edx, %edx
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
.LBB163_7:
	movl	$1336, %esi
	leaq	32(%rsp), %r8
	movq	%rax, 32(%rsp)
	movq	%rdx, 40(%rsp)
	movq	%r13, 48(%rsp)
	bextrq	%rsi, %r13, %rcx
	testq	%r13, %r13
	cmovnsq	%rax, %r8
	movq	24(%rsp), %rax
	cmovnsq	%rdx, %rcx
	testq	%rax, %rax
	js	.LBB163_8
	movq	8(%rsp), %rdi
	movq	16(%rsp), %rsi
	jmp	.LBB163_10
.LBB163_8:
	bextrq	%rsi, %rax, %rsi
	leaq	8(%rsp), %rdi
.LBB163_10:
	movq	%r8, %rdx
	movabsq	$4611686018427387904, %r13
	callq	"std::collections::string::string::String::_add[::Bool,LITOrigin[$0._mlir_value],::Origin[$0, $1],::Bool,LITOrigin[$3._mlir_value],::Origin[$3, $4]](::Span[$0, $1, ::SIMD[DType.uint8, 1], $2, AddressSpace.GENERIC],::Span[$3, $4, ::SIMD[DType.uint8, 1], $5, AddressSpace.GENERIC]),lhs.mut`2x=false,rhs.mut`2x3=false"@PLT
	testb	$64, 55(%rsp)
	movq	%rax, 56(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rcx, 72(%rsp)
	je	.LBB163_13
	movq	32(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB163_13
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB163_13:
	movq	160(%rsp), %rcx
	testq	%r13, 24(%rsp)
	je	.LBB163_16
	movq	8(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB163_16
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rcx, %rbp
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbp, %rcx
.LBB163_16:
	leaq	56(%rsp), %r8
	leaq	32(%rsp), %r9
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	%r14, %rdx
	callq	"std::reflection::location::SourceLocation::prefix[::Writable & ::AnyType](::SourceLocation,$0),T=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	testq	%r13, 72(%rsp)
	je	.LBB163_19
	movq	56(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB163_19
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB163_19:
	vmovups	40(%rsp), %xmm0
	movq	32(%rsp), %r14
	movq	$-1, %rdi
	vmovaps	%xmm0, 80(%rsp)
	callq	"std::builtin::error::StackTrace::collect_if_enabled(::SIMD[DType.int, 1])"@PLT
	movb	%dl, 40(%rbx)
	movq	%rax, 32(%rbx)
	movq	%rbx, %rax
	vmovaps	80(%rsp), %xmm0
	vmovups	%xmm0, 16(%rbx)
	movq	%r14, 8(%rbx)
	movb	$1, (%rbx)
	addq	$104, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end30:
	.size	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG", .Lfunc_end30-"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"
	.cfi_endproc

	.prefalign	4, .Lfunc_end31, nop
	.type	"std::io::io::print[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2],::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5],::Bool,LITOrigin[$7._mlir_value],::Origin[$7, $8]](*$0,sep:::StringSpan[$4, $5, $6],end:::StringSpan[$7, $8, $9],flush:::Bool,file:::FileDescriptor$)_REMOVED_ARG,Ts.values`=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]",@function
"std::io::io::print[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2],::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5],::Bool,LITOrigin[$7._mlir_value],::Origin[$7, $8]](*$0,sep:::StringSpan[$4, $5, $6],end:::StringSpan[$7, $8, $9],flush:::Bool,file:::FileDescriptor$)_REMOVED_ARG,Ts.values`=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4200, %rsp
	.cfi_def_cfa_offset 4256
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%r9, 40(%rsp)
	movq	%r8, 32(%rsp)
	movq	%rcx, 24(%rsp)
	leaq	16(%rsp), %rcx
	leaq	88(%rsp), %rbx
	movq	%rdx, %r15
	movq	%rsi, %rdx
	movq	%rdi, %rsi
	movq	4320(%rsp), %rax
	movq	%rbx, %rdi
	movq	%rax, 80(%rsp)
	movq	4312(%rsp), %rax
	movq	%rax, 72(%rsp)
	movq	4280(%rsp), %rax
	movq	4304(%rsp), %r12
	movq	4296(%rsp), %r13
	movq	%rax, 56(%rsp)
	movq	4272(%rsp), %rax
	movq	%rax, 48(%rsp)
	movzbl	4328(%rsp), %eax
	movb	%al, 15(%rsp)
	movq	4288(%rsp), %rax
	movq	%rax, 64(%rsp)
	movq	4336(%rsp), %rax
	movq	4256(%rsp), %r14
	movq	4264(%rsp), %rbp
	movq	$0, 4184(%rsp)
	movq	%rcx, 4192(%rsp)
	movq	%rax, 16(%rsp)
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	%rbx, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	24(%rsp), %rdx
	movq	%rbx, %rdi
	movq	%r15, %rsi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	%rbx, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	%rbx, %rdi
	movq	%r14, %rsi
	movq	%rbp, %rdx
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4184(%rsp), %rdx
	leaq	1(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB164_2
	movq	4192(%rsp), %rax
	leaq	88(%rsp), %rsi
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%edx, %edx
	movq	$0, 4184(%rsp)
.LBB164_2:
	movb	$58, 88(%rsp,%rdx)
	incq	4184(%rsp)
	movq	32(%rsp), %rdi
	leaq	88(%rsp), %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4184(%rsp), %rdx
	leaq	1(%rdx), %rax
	cmpq	$4097, %rax
	jl	.LBB164_4
	movq	4192(%rsp), %rax
	leaq	88(%rsp), %rsi
	movq	(%rax), %rdi
	callq	write@PLT
	xorl	%edx, %edx
	movq	$0, 4184(%rsp)
.LBB164_4:
	movb	$58, 88(%rsp,%rdx)
	incq	4184(%rsp)
	movq	40(%rsp), %rdi
	leaq	88(%rsp), %rbx
	movq	%rbx, %rsi
	callq	"std::simd::SIMD::write_to[::Writer & ::AnyType](::SIMD[$0, $1],$2),dtype=index,length=1,writer.T`2x=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	%rbx, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	48(%rsp), %rsi
	movq	56(%rsp), %rdx
	movq	%rbx, %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	%rbx, %rdi
	movq	%r13, %rsi
	movq	%r12, %rdx
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	64(%rsp), %rsi
	movq	16(%rsi), %rax
	testq	%rax, %rax
	js	.LBB164_5
	movq	8(%rsi), %rdx
	movq	(%rsi), %rsi
	jmp	.LBB164_7
.LBB164_5:
	movl	$1336, %ecx
	bextrq	%rcx, %rax, %rdx
.LBB164_7:
	leaq	88(%rsp), %rbx
	movq	%rbx, %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	72(%rsp), %rsi
	movq	80(%rsp), %rdx
	movq	%rbx, %rdi
	callq	"std::format::_utils::_FlushingWriteBuffer::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_FlushingWriteBuffer[$0, $1, $2, $3],::StringSpan[$4, $5, $6]),W=[typevalue<#kgen.instref<std::io::file_descriptor::FileDescriptor>>, scalar<index>],capacity_bytes=4096,string.mut`2x1=false"@PLT
	movq	4192(%rsp), %rax
	movq	4184(%rsp), %rdx
	movq	%rbx, %rsi
	movq	(%rax), %rdi
	callq	write@PLT
	testb	$1, 15(%rsp)
	je	.LBB164_8
	movq	16(%rsp), %rdi
	addq	$4200, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmp	"std::io::io::_flush(::FileDescriptor)"@PLT
.LBB164_8:
	.cfi_def_cfa_offset 4256
	addq	$4200, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end31:
	.size	"std::io::io::print[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2],::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5],::Bool,LITOrigin[$7._mlir_value],::Origin[$7, $8]](*$0,sep:::StringSpan[$4, $5, $6],end:::StringSpan[$7, $8, $9],flush:::Bool,file:::FileDescriptor$)_REMOVED_ARG,Ts.values`=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]", .Lfunc_end31-"std::io::io::print[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2],::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5],::Bool,LITOrigin[$7._mlir_value],::Origin[$7, $8]](*$0,sep:::StringSpan[$4, $5, $6],end:::StringSpan[$7, $8, $9],flush:::Bool,file:::FileDescriptor$)_REMOVED_ARG,Ts.values`=[[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"
	.cfi_endproc

	.globl	gate_lds
	.prefalign	4, .Lfunc_end32, nop
	.type	gate_lds,@function
gate_lds:
.Lgate_lds$local:
	.type	.Lgate_lds$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4728, %rsp
	.cfi_def_cfa_offset 4784
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 16(%rsp)
	leaq	static_string_3b45d8d0deb8f5b3(%rip), %rax
	leaq	208(%rsp), %rbx
	movl	$4, %esi
	movq	%rdi, 192(%rsp)
	movabsq	$2305843009213693952, %rbp
	movq	$0, 48(%rsp)
	movq	%rax, 208(%rsp)
	movq	$3, 216(%rsp)
	movq	%rbx, %rdi
	movq	$0, 224(%rsp)
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB268_2
	movq	216(%rsp), %rcx
	jmp	.LBB268_3
.LBB268_2:
	movl	$1336, %edx
	bextrq	%rdx, %rcx, %rcx
.LBB268_3:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	movq	%rax, %rcx
	orq	%rbp, %rcx
	movq	%rcx, 224(%rsp)
	testq	%rax, %rax
	js	.LBB268_5
	movq	208(%rsp), %rbx
.LBB268_5:
	leaq	48(%rsp), %rdi
	movq	%rbx, %rsi
	xorl	%edx, %edx
	callq	AsyncRT_DeviceContext_create@PLT
	testb	$64, 231(%rsp)
	je	.LBB268_8
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_8
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %rbx
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rax
.LBB268_8:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movabsq	$4611686018427387904, %r15
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB268_18
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_69b8ed51c73424d7(%rip), %r9
	leaq	536(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$4010, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$41
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	%r15, %rax
	movq	536(%rsp), %rbx
	movq	552(%rsp), %r13
	movq	560(%rsp), %r14
	movzbl	568(%rsp), %r12d
	movzbl	528(%rsp), %r15d
	movq	%rax, %rbp
	testq	%rax, 632(%rsp)
	je	.LBB268_12
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_12
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_12:
	testb	$1, %r15b
	movq	%rbp, %r15
	movabsq	$2305843009213693952, %rbp
	je	.LBB268_18
	testq	%r15, %r13
	je	.LBB268_16
	lock		decq	-8(%rbx)
	jne	.LBB268_16
	addq	$-8, %rbx
	#MEMBARRIER
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_16:
	movl	$1, %ebx
	cmpb	$1, %r12b
	jne	.LBB268_84
	movq	%r14, %rdi
	jmp	.LBB268_83
.LBB268_18:
	movq	48(%rsp), %rbx
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_retain@PLT
	xorl	%edi, %edi
	movq	$0, 48(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	cmpq	$23, %rax
	jg	.LBB268_22
	leaq	208(%rsp), %rdi
	movabsq	$-9223372036854775808, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, 208(%rsp)
	movq	%rax, 224(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movq	224(%rsp), %rax
	testq	%rbp, %rax
	jne	.LBB268_20
.LBB268_27:
	movl	$1336, %r14d
	testq	%rax, %rax
	js	.LBB268_29
	movq	216(%rsp), %rsi
	jmp	.LBB268_30
.LBB268_22:
	movq	%rax, %r14
	addq	$7, %r14
	movq	%r14, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB268_98
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB268_100
	sarq	$3, %r14
	movq	$1, (%rax)
	addq	$8, %rax
	movq	$0, 4712(%rsp)
	orq	%r15, %r14
	movq	%rax, 208(%rsp)
	movq	$0, 216(%rsp)
	leaq	208(%rsp), %rax
	movq	%r14, 224(%rsp)
	leaq	616(%rsp), %r14
	movq	%rax, 4720(%rsp)
	movq	%r14, %rdi
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4712(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB268_26
	movq	4720(%rsp), %rdi
	leaq	616(%rsp), %r14
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4712(%rsp)
.LBB268_26:
	movq	4720(%rsp), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	224(%rsp), %rax
	testq	%rbp, %rax
	je	.LBB268_27
.LBB268_20:
	testq	%rax, %rax
	js	.LBB268_34
.LBB268_21:
	movq	208(%rsp), %rax
	jmp	.LBB268_35
.LBB268_29:
	bextrq	%r14, %rax, %rsi
.LBB268_30:
	incq	%rsi
	leaq	208(%rsp), %rdi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB268_32
	movq	216(%rsp), %rcx
	jmp	.LBB268_33
.LBB268_32:
	bextrq	%r14, %rcx, %rcx
.LBB268_33:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	orq	%rbp, %rax
	movq	%rax, 224(%rsp)
	testq	%rax, %rax
	jns	.LBB268_21
.LBB268_34:
	leaq	208(%rsp), %rax
.LBB268_35:
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_503eabc945fed7e4(%rip), %rdx
	leaq	static_string_fad8c1e7bc02fc5f(%rip), %rcx
	leaq	static_string_938a41beea320a7b(%rip), %r8
	leaq	56(%rsp), %rdi
	movl	$3768, %r9d
	movq	%rbx, %rsi
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$-1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_loadFunction@PLT
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	testq	%r15, 224(%rsp)
	je	.LBB268_38
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_38
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r14, %rax
.LBB268_38:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB268_45
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f6c401c515dc2a68(%rip), %r9
	leaq	488(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$170, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$49
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	488(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	512(%rsp), %rax
	movq	504(%rsp), %r13
	movq	%rax, 88(%rsp)
	movq	%r15, %rax
	movq	%rax, %r14
	movzbl	520(%rsp), %r12d
	movzbl	480(%rsp), %r15d
	testq	%rax, 632(%rsp)
	je	.LBB268_43
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_43
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testb	$1, %r15b
	jne	.LBB268_44
.LBB268_42:
	movq	%r14, %r15
	movq	48(%rsp), %r14
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	jne	.LBB268_46
.LBB268_49:
	leaq	592(%rsp), %rcx
	leaq	599(%rsp), %rax
	movq	%r14, 8(%rsp)
	leaq	616(%rsp), %r13
	movq	$1, 616(%rsp)
	movq	$0, 624(%rsp)
	testq	%rcx, %rcx
	movq	%r13, %rdi
	cmovnsq	%rcx, %rax
	setns	%dl
	sarq	$3, %rax
	movq	%rax, %rsi
	negq	%rsi
	shlq	$3, %rsi
	addq	%rcx, %rsi
	movq	%rbx, %rsi
	setne	%cl
	andb	%dl, %cl
	movzbl	%cl, %r14d
	addq	%rax, %r14
	shlq	$3, %r14
	leaq	8(%r14), %r12
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	movq	624(%rsp), %rdx
	cmpq	$5, %rdx
	jne	.LBB268_54
	movq	616(%rsp), %rax
	leaq	static_string_561d9efdd15f277f(%rip), %rcx
	cmpq	%rcx, %rax
	je	.LBB268_56
	movq	%rdx, %rsi
	sarq	$63, %rsi
	andnq	%rdx, %rsi, %rdx
	xorl	%esi, %esi
	.p2align	4
.LBB268_52:
	cmpq	%rsi, %rdx
	je	.LBB268_56
	movzbl	(%rax,%rsi), %edi
	cmpb	(%rsi,%rcx), %dil
	leaq	1(%rsi), %rsi
	je	.LBB268_52
.LBB268_54:
	movq	192(%rsp), %rax
	movq	16(%rsp), %rsi
	movq	%r14, 208(%rsp)
	movq	%r12, 216(%rsp)
	movq	$0, 576(%rsp)
	movq	%rbx, %rdi
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%rax, (%r14)
	movq	%rsi, 8(%r14)
	movq	8(%rsp), %r14
	leaq	208(%rsp), %rax
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$2
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB268_85
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	616(%rsp), %r13
	movq	%rax, 96(%rsp)
	movq	%rdx, 104(%rsp)
	movq	$1, 616(%rsp)
	movq	%rcx, 112(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbx, %rsi
	movq	%r13, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	616(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 168(%rsp)
	movq	$0, 184(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 128(%rsp)
	movq	$12, 152(%rsp)
	movq	$1, 32(%rsp)
	movq	$13, 56(%rsp)
	movq	$1, 624(%rsp)
	movq	%rcx, 120(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 136(%rsp)
	movq	%rcx, 144(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 160(%rsp)
	movq	%rcx, 24(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 40(%rsp)
	movq	%rcx, 48(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 64(%rsp)
	movq	%rcx, 616(%rsp)
	movq	%rbp, 632(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	152(%rsp), %r12
	leaq	static_string_f1dde65067a8a2a0(%rip), %rbp
	leaq	104(%rsp), %r10
	leaq	32(%rsp), %r11
	leaq	176(%rsp), %r14
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	448(%rsp), %rdi
	leaq	128(%rsp), %r9
	movl	$114, %esi
	movl	$41, %edx
	movl	$25, %r8d
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	leaq	72(%rsp), %r10
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$624
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movq	464(%rsp), %rax
	movzbl	472(%rsp), %ebp
	movq	%rax, 88(%rsp)
	movq	456(%rsp), %r12
	movq	440(%rsp), %r13
	testq	%r15, 632(%rsp)
	jne	.LBB268_58
	jmp	.LBB268_60
.LBB268_43:
	testb	$1, %r15b
	je	.LBB268_42
.LBB268_44:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %r15
	jmp	.LBB268_47
.LBB268_45:
	movq	48(%rsp), %r14
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	je	.LBB268_49
.LBB268_46:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
.LBB268_47:
	movl	%r12d, %ebp
	movq	%r13, %r12
	movq	8(%rsp), %r13
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %r12
	jne	.LBB268_79
	jmp	.LBB268_81
.LBB268_56:
	movq	192(%rsp), %rax
	movq	16(%rsp), %rsi
	movq	%r14, 208(%rsp)
	movq	%r12, 216(%rsp)
	leaq	48(%rsp), %r12
	vxorps	%xmm0, %xmm0, %xmm0
	vmovups	%xmm0, 26(%rsp)
	movb	$0, 24(%rsp)
	vmovups	%zmm0, 696(%rsp)
	vmovups	%zmm0, 632(%rsp)
	movq	$8, 616(%rsp)
	movq	$8, 624(%rsp)
	movb	$0, 25(%rsp)
	movq	$0, 584(%rsp)
	movq	%rbx, %rdi
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%r12, 200(%rsp)
	movq	%rax, (%r14)
	movq	%rsi, 8(%r14)
	movq	8(%rsp), %r14
	leaq	208(%rsp), %rax
	movq	%rax, 48(%rsp)
	movq	%r13, 56(%rsp)
	leaq	24(%rsp), %r13
	leaq	200(%rsp), %rax
	movq	%r13, 64(%rsp)
	movq	$8, 72(%rsp)
	movl	$0, 80(%rsp)
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$2
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	vzeroupper
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB268_85
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	616(%rsp), %r14
	movq	%rax, 96(%rsp)
	movq	%rdx, 104(%rsp)
	movq	$1, 616(%rsp)
	movq	%rcx, 112(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	616(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 168(%rsp)
	movq	$0, 184(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 128(%rsp)
	movq	$12, 152(%rsp)
	movq	$1, 32(%rsp)
	movq	$13, 56(%rsp)
	movq	$1, 624(%rsp)
	movq	%rcx, 120(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 136(%rsp)
	movq	%rcx, 144(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 160(%rsp)
	movq	%rcx, 24(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 40(%rsp)
	movq	%rcx, 48(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 64(%rsp)
	movq	%rcx, 616(%rsp)
	movq	%rbp, 632(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movq	%r12, %rbp
	leaq	152(%rsp), %r12
	leaq	104(%rsp), %r10
	leaq	176(%rsp), %r11
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	408(%rsp), %rdi
	leaq	128(%rsp), %r9
	movl	$114, %esi
	movl	$41, %edx
	movl	$25, %r8d
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$624
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f1dde65067a8a2a0(%rip), %rax
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movq	424(%rsp), %rax
	movzbl	432(%rsp), %ebp
	movq	%rax, 88(%rsp)
	movq	416(%rsp), %r12
	movq	400(%rsp), %r13
	testq	%r15, 632(%rsp)
	je	.LBB268_60
.LBB268_58:
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_60
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_60:
	testq	%r15, 64(%rsp)
	je	.LBB268_63
	movq	48(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_63
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_63:
	movq	8(%rsp), %r14
	testq	%r15, 40(%rsp)
	je	.LBB268_66
	movq	24(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_66
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_66:
	testq	%r15, 160(%rsp)
	je	.LBB268_69
	movq	144(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_69
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_69:
	testq	%r15, 136(%rsp)
	je	.LBB268_72
	movq	120(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_72
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_72:
	testq	%r15, 184(%rsp)
	je	.LBB268_75
	movq	168(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_75
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_75:
	testq	%r15, 112(%rsp)
	je	.LBB268_78
	movq	96(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_78
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_78:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %r12
	je	.LBB268_81
.LBB268_79:
	lock		decq	-8(%r13)
	jne	.LBB268_81
	addq	$-8, %r13
	#MEMBARRIER
	movq	%r13, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_81:
	movl	$1, %ebx
	cmpb	$1, %bpl
	jne	.LBB268_84
	movq	88(%rsp), %rdi
.LBB268_83:
	callq	"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]"@PLT
.LBB268_84:
	movq	%rbx, %rax
	addq	$4728, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB268_85:
	.cfi_def_cfa_offset 4784
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_synchronize@PLT
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB268_96
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f078bd8d2bcbf530(%rip), %r9
	leaq	360(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$115, %ecx
	movl	$24, %r8d
	movq	%rax, %rsi
	pushq	$25
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	384(%rsp), %rcx
	movq	%r15, %rax
	movq	360(%rsp), %r15
	movq	376(%rsp), %r12
	movq	%rax, %r14
	movq	%rcx, 16(%rsp)
	movzbl	392(%rsp), %ebp
	movzbl	352(%rsp), %r13d
	testq	%rax, 632(%rsp)
	je	.LBB268_90
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB268_90
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	jne	.LBB268_91
	jmp	.LBB268_89
.LBB268_90:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	je	.LBB268_89
.LBB268_91:
	testq	%r14, %r12
	je	.LBB268_94
	lock		decq	-8(%r15)
	jne	.LBB268_94
	addq	$-8, %r15
	#MEMBARRIER
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB268_94:
	movl	$1, %ebx
	cmpb	$1, %bpl
	jne	.LBB268_84
	movq	16(%rsp), %rdi
	jmp	.LBB268_83
.LBB268_96:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
.LBB268_89:
	xorl	%ebx, %ebx
	jmp	.LBB268_84
.LBB268_98:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	616(%rsp), %r8
	movl	$783, %edi
	movq	$35, 624(%rsp)
	jmp	.LBB268_99
.LBB268_100:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movq	$37, 624(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	616(%rsp), %r8
	movl	$659, %edi
.LBB268_99:
	movq	%rax, 616(%rsp)
	movq	%rbp, 632(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end32:
	.size	gate_lds, .Lfunc_end32-gate_lds
	.size	.Lgate_lds$local, .Lfunc_end32-gate_lds
	.cfi_endproc

	.globl	gate_nt
	.prefalign	4, .Lfunc_end33, nop
	.type	gate_nt,@function
gate_nt:
.Lgate_nt$local:
	.type	.Lgate_nt$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4728, %rsp
	.cfi_def_cfa_offset 4784
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 16(%rsp)
	leaq	static_string_3b45d8d0deb8f5b3(%rip), %rax
	leaq	208(%rsp), %rbx
	movl	$4, %esi
	movq	%rdi, 192(%rsp)
	movabsq	$2305843009213693952, %rbp
	movq	$0, 48(%rsp)
	movq	%rax, 208(%rsp)
	movq	$3, 216(%rsp)
	movq	%rbx, %rdi
	movq	$0, 224(%rsp)
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB402_2
	movq	216(%rsp), %rcx
	jmp	.LBB402_3
.LBB402_2:
	movl	$1336, %edx
	bextrq	%rdx, %rcx, %rcx
.LBB402_3:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	movq	%rax, %rcx
	orq	%rbp, %rcx
	movq	%rcx, 224(%rsp)
	testq	%rax, %rax
	js	.LBB402_5
	movq	208(%rsp), %rbx
.LBB402_5:
	leaq	48(%rsp), %rdi
	movq	%rbx, %rsi
	xorl	%edx, %edx
	callq	AsyncRT_DeviceContext_create@PLT
	testb	$64, 231(%rsp)
	je	.LBB402_8
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_8
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %rbx
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rax
.LBB402_8:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movabsq	$4611686018427387904, %r15
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB402_18
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_69b8ed51c73424d7(%rip), %r9
	leaq	536(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$4010, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$41
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	%r15, %rax
	movq	536(%rsp), %rbx
	movq	552(%rsp), %r13
	movq	560(%rsp), %r14
	movzbl	568(%rsp), %r12d
	movzbl	528(%rsp), %r15d
	movq	%rax, %rbp
	testq	%rax, 632(%rsp)
	je	.LBB402_12
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_12
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_12:
	testb	$1, %r15b
	movq	%rbp, %r15
	movabsq	$2305843009213693952, %rbp
	je	.LBB402_18
	testq	%r15, %r13
	je	.LBB402_16
	lock		decq	-8(%rbx)
	jne	.LBB402_16
	addq	$-8, %rbx
	#MEMBARRIER
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_16:
	movl	$1, %ebx
	cmpb	$1, %r12b
	jne	.LBB402_84
	movq	%r14, %rdi
	jmp	.LBB402_83
.LBB402_18:
	movq	48(%rsp), %rbx
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_retain@PLT
	xorl	%edi, %edi
	movq	$0, 48(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	cmpq	$23, %rax
	jg	.LBB402_22
	leaq	208(%rsp), %rdi
	movabsq	$-9223372036854775808, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, 208(%rsp)
	movq	%rax, 224(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movq	224(%rsp), %rax
	testq	%rbp, %rax
	jne	.LBB402_20
.LBB402_27:
	movl	$1336, %r14d
	testq	%rax, %rax
	js	.LBB402_29
	movq	216(%rsp), %rsi
	jmp	.LBB402_30
.LBB402_22:
	movq	%rax, %r14
	addq	$7, %r14
	movq	%r14, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB402_98
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB402_100
	sarq	$3, %r14
	movq	$1, (%rax)
	addq	$8, %rax
	movq	$0, 4712(%rsp)
	orq	%r15, %r14
	movq	%rax, 208(%rsp)
	movq	$0, 216(%rsp)
	leaq	208(%rsp), %rax
	movq	%r14, 224(%rsp)
	leaq	616(%rsp), %r14
	movq	%rax, 4720(%rsp)
	movq	%r14, %rdi
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4712(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB402_26
	movq	4720(%rsp), %rdi
	leaq	616(%rsp), %r14
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4712(%rsp)
.LBB402_26:
	movq	4720(%rsp), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	224(%rsp), %rax
	testq	%rbp, %rax
	je	.LBB402_27
.LBB402_20:
	testq	%rax, %rax
	js	.LBB402_34
.LBB402_21:
	movq	208(%rsp), %rax
	jmp	.LBB402_35
.LBB402_29:
	bextrq	%r14, %rax, %rsi
.LBB402_30:
	incq	%rsi
	leaq	208(%rsp), %rdi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB402_32
	movq	216(%rsp), %rcx
	jmp	.LBB402_33
.LBB402_32:
	bextrq	%r14, %rcx, %rcx
.LBB402_33:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	orq	%rbp, %rax
	movq	%rax, 224(%rsp)
	testq	%rax, %rax
	jns	.LBB402_21
.LBB402_34:
	leaq	208(%rsp), %rax
.LBB402_35:
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_fe5419fd1e6ae11d(%rip), %rdx
	leaq	static_string_9aef310a067be18e(%rip), %rcx
	leaq	static_string_707259faf29623c9(%rip), %r8
	leaq	56(%rsp), %rdi
	movl	$3512, %r9d
	movq	%rbx, %rsi
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$-1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_loadFunction@PLT
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	testq	%r15, 224(%rsp)
	je	.LBB402_38
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_38
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r14, %rax
.LBB402_38:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB402_45
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f6c401c515dc2a68(%rip), %r9
	leaq	488(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$170, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$49
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	488(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	512(%rsp), %rax
	movq	504(%rsp), %r13
	movq	%rax, 88(%rsp)
	movq	%r15, %rax
	movq	%rax, %r14
	movzbl	520(%rsp), %r12d
	movzbl	480(%rsp), %r15d
	testq	%rax, 632(%rsp)
	je	.LBB402_43
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_43
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testb	$1, %r15b
	jne	.LBB402_44
.LBB402_42:
	movq	%r14, %r15
	movq	48(%rsp), %r14
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	jne	.LBB402_46
.LBB402_49:
	leaq	592(%rsp), %rcx
	leaq	599(%rsp), %rax
	movq	%r14, 8(%rsp)
	leaq	616(%rsp), %r13
	movq	$1, 616(%rsp)
	movq	$0, 624(%rsp)
	testq	%rcx, %rcx
	movq	%r13, %rdi
	cmovnsq	%rcx, %rax
	setns	%dl
	sarq	$3, %rax
	movq	%rax, %rsi
	negq	%rsi
	shlq	$3, %rsi
	addq	%rcx, %rsi
	movq	%rbx, %rsi
	setne	%cl
	andb	%dl, %cl
	movzbl	%cl, %r14d
	addq	%rax, %r14
	shlq	$3, %r14
	leaq	8(%r14), %r12
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	movq	624(%rsp), %rdx
	cmpq	$5, %rdx
	jne	.LBB402_54
	movq	616(%rsp), %rax
	leaq	static_string_561d9efdd15f277f(%rip), %rcx
	cmpq	%rcx, %rax
	je	.LBB402_56
	movq	%rdx, %rsi
	sarq	$63, %rsi
	andnq	%rdx, %rsi, %rdx
	xorl	%esi, %esi
	.p2align	4
.LBB402_52:
	cmpq	%rsi, %rdx
	je	.LBB402_56
	movzbl	(%rax,%rsi), %edi
	cmpb	(%rsi,%rcx), %dil
	leaq	1(%rsi), %rsi
	je	.LBB402_52
.LBB402_54:
	movq	192(%rsp), %rax
	movq	16(%rsp), %rsi
	movq	%r14, 208(%rsp)
	movq	%r12, 216(%rsp)
	movq	$0, 576(%rsp)
	movq	%rbx, %rdi
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%rax, (%r14)
	movq	%rsi, 8(%r14)
	movq	8(%rsp), %r14
	leaq	208(%rsp), %rax
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$2
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB402_85
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	616(%rsp), %r13
	movq	%rax, 96(%rsp)
	movq	%rdx, 104(%rsp)
	movq	$1, 616(%rsp)
	movq	%rcx, 112(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbx, %rsi
	movq	%r13, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	616(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 168(%rsp)
	movq	$0, 184(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 128(%rsp)
	movq	$12, 152(%rsp)
	movq	$1, 32(%rsp)
	movq	$13, 56(%rsp)
	movq	$1, 624(%rsp)
	movq	%rcx, 120(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 136(%rsp)
	movq	%rcx, 144(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 160(%rsp)
	movq	%rcx, 24(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 40(%rsp)
	movq	%rcx, 48(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 64(%rsp)
	movq	%rcx, 616(%rsp)
	movq	%rbp, 632(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	152(%rsp), %r12
	leaq	static_string_f1dde65067a8a2a0(%rip), %rbp
	leaq	104(%rsp), %r10
	leaq	32(%rsp), %r11
	leaq	176(%rsp), %r14
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	448(%rsp), %rdi
	leaq	128(%rsp), %r9
	movl	$125, %esi
	movl	$40, %edx
	movl	$25, %r8d
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	leaq	72(%rsp), %r10
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$624
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movq	464(%rsp), %rax
	movzbl	472(%rsp), %ebp
	movq	%rax, 88(%rsp)
	movq	456(%rsp), %r12
	movq	440(%rsp), %r13
	testq	%r15, 632(%rsp)
	jne	.LBB402_58
	jmp	.LBB402_60
.LBB402_43:
	testb	$1, %r15b
	je	.LBB402_42
.LBB402_44:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %r15
	jmp	.LBB402_47
.LBB402_45:
	movq	48(%rsp), %r14
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	je	.LBB402_49
.LBB402_46:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
.LBB402_47:
	movl	%r12d, %ebp
	movq	%r13, %r12
	movq	8(%rsp), %r13
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %r12
	jne	.LBB402_79
	jmp	.LBB402_81
.LBB402_56:
	movq	192(%rsp), %rax
	movq	16(%rsp), %rsi
	movq	%r14, 208(%rsp)
	movq	%r12, 216(%rsp)
	leaq	48(%rsp), %r12
	vxorps	%xmm0, %xmm0, %xmm0
	vmovups	%xmm0, 26(%rsp)
	movb	$0, 24(%rsp)
	vmovups	%zmm0, 696(%rsp)
	vmovups	%zmm0, 632(%rsp)
	movq	$8, 616(%rsp)
	movq	$8, 624(%rsp)
	movb	$0, 25(%rsp)
	movq	$0, 584(%rsp)
	movq	%rbx, %rdi
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%r12, 200(%rsp)
	movq	%rax, (%r14)
	movq	%rsi, 8(%r14)
	movq	8(%rsp), %r14
	leaq	208(%rsp), %rax
	movq	%rax, 48(%rsp)
	movq	%r13, 56(%rsp)
	leaq	24(%rsp), %r13
	leaq	200(%rsp), %rax
	movq	%r13, 64(%rsp)
	movq	$8, 72(%rsp)
	movl	$0, 80(%rsp)
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$2
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	vzeroupper
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB402_85
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	616(%rsp), %r14
	movq	%rax, 96(%rsp)
	movq	%rdx, 104(%rsp)
	movq	$1, 616(%rsp)
	movq	%rcx, 112(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	616(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 168(%rsp)
	movq	$0, 184(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 128(%rsp)
	movq	$12, 152(%rsp)
	movq	$1, 32(%rsp)
	movq	$13, 56(%rsp)
	movq	$1, 624(%rsp)
	movq	%rcx, 120(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 136(%rsp)
	movq	%rcx, 144(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 160(%rsp)
	movq	%rcx, 24(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 40(%rsp)
	movq	%rcx, 48(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 64(%rsp)
	movq	%rcx, 616(%rsp)
	movq	%rbp, 632(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	movq	%r12, %rbp
	leaq	152(%rsp), %r12
	leaq	104(%rsp), %r10
	leaq	176(%rsp), %r11
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	408(%rsp), %rdi
	leaq	128(%rsp), %r9
	movl	$125, %esi
	movl	$40, %edx
	movl	$25, %r8d
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$624
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f1dde65067a8a2a0(%rip), %rax
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movq	424(%rsp), %rax
	movzbl	432(%rsp), %ebp
	movq	%rax, 88(%rsp)
	movq	416(%rsp), %r12
	movq	400(%rsp), %r13
	testq	%r15, 632(%rsp)
	je	.LBB402_60
.LBB402_58:
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_60
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_60:
	testq	%r15, 64(%rsp)
	je	.LBB402_63
	movq	48(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_63
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_63:
	movq	8(%rsp), %r14
	testq	%r15, 40(%rsp)
	je	.LBB402_66
	movq	24(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_66
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_66:
	testq	%r15, 160(%rsp)
	je	.LBB402_69
	movq	144(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_69
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_69:
	testq	%r15, 136(%rsp)
	je	.LBB402_72
	movq	120(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_72
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_72:
	testq	%r15, 184(%rsp)
	je	.LBB402_75
	movq	168(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_75
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_75:
	testq	%r15, 112(%rsp)
	je	.LBB402_78
	movq	96(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_78
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_78:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %r12
	je	.LBB402_81
.LBB402_79:
	lock		decq	-8(%r13)
	jne	.LBB402_81
	addq	$-8, %r13
	#MEMBARRIER
	movq	%r13, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_81:
	movl	$1, %ebx
	cmpb	$1, %bpl
	jne	.LBB402_84
	movq	88(%rsp), %rdi
.LBB402_83:
	callq	"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]"@PLT
.LBB402_84:
	movq	%rbx, %rax
	addq	$4728, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB402_85:
	.cfi_def_cfa_offset 4784
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_synchronize@PLT
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 616(%rsp)
	movq	$0, 624(%rsp)
	movq	%rbp, 632(%rsp)
	testq	%rax, %rax
	je	.LBB402_96
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f078bd8d2bcbf530(%rip), %r9
	leaq	360(%rsp), %rdi
	leaq	624(%rsp), %rdx
	movl	$126, %ecx
	movl	$24, %r8d
	movq	%rax, %rsi
	pushq	$25
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	384(%rsp), %rcx
	movq	%r15, %rax
	movq	360(%rsp), %r15
	movq	376(%rsp), %r12
	movq	%rax, %r14
	movq	%rcx, 16(%rsp)
	movzbl	392(%rsp), %ebp
	movzbl	352(%rsp), %r13d
	testq	%rax, 632(%rsp)
	je	.LBB402_90
	movq	616(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB402_90
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	jne	.LBB402_91
	jmp	.LBB402_89
.LBB402_90:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	je	.LBB402_89
.LBB402_91:
	testq	%r14, %r12
	je	.LBB402_94
	lock		decq	-8(%r15)
	jne	.LBB402_94
	addq	$-8, %r15
	#MEMBARRIER
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB402_94:
	movl	$1, %ebx
	cmpb	$1, %bpl
	jne	.LBB402_84
	movq	16(%rsp), %rdi
	jmp	.LBB402_83
.LBB402_96:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
.LBB402_89:
	xorl	%ebx, %ebx
	jmp	.LBB402_84
.LBB402_98:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	616(%rsp), %r8
	movl	$783, %edi
	movq	$35, 624(%rsp)
	jmp	.LBB402_99
.LBB402_100:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movq	$37, 624(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	616(%rsp), %r8
	movl	$659, %edi
.LBB402_99:
	movq	%rax, 616(%rsp)
	movq	%rbp, 632(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end33:
	.size	gate_nt, .Lfunc_end33-gate_nt
	.size	.Lgate_nt$local, .Lfunc_end33-gate_nt
	.cfi_endproc

	.globl	gate_dot2
	.prefalign	4, .Lfunc_end34, nop
	.type	gate_dot2,@function
gate_dot2:
.Lgate_dot2$local:
	.type	.Lgate_dot2$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4744, %rsp
	.cfi_def_cfa_offset 4800
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 184(%rsp)
	leaq	static_string_3b45d8d0deb8f5b3(%rip), %rax
	leaq	208(%rsp), %rbx
	movl	$4, %esi
	movq	%rdi, 176(%rsp)
	movq	%rdx, 192(%rsp)
	movabsq	$2305843009213693952, %r13
	movq	$0, 40(%rsp)
	movq	%rax, 208(%rsp)
	movq	$3, 216(%rsp)
	movq	%rbx, %rdi
	movq	$0, 224(%rsp)
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB536_2
	movq	216(%rsp), %rcx
	jmp	.LBB536_3
.LBB536_2:
	movl	$1336, %edx
	bextrq	%rdx, %rcx, %rcx
.LBB536_3:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	movq	%rax, %rcx
	orq	%r13, %rcx
	movq	%rcx, 224(%rsp)
	testq	%rax, %rax
	js	.LBB536_5
	movq	208(%rsp), %rbx
.LBB536_5:
	leaq	40(%rsp), %rdi
	movq	%rbx, %rsi
	xorl	%edx, %edx
	callq	AsyncRT_DeviceContext_create@PLT
	testb	$64, 231(%rsp)
	je	.LBB536_8
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_8
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %rbx
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rax
.LBB536_8:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movabsq	$4611686018427387904, %rbp
	movq	%rcx, 632(%rsp)
	movq	$0, 640(%rsp)
	movq	%r13, 648(%rsp)
	testq	%rax, %rax
	je	.LBB536_18
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_69b8ed51c73424d7(%rip), %r9
	leaq	544(%rsp), %rdi
	leaq	640(%rsp), %rdx
	movl	$4010, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$41
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	544(%rsp), %rbx
	movq	560(%rsp), %r12
	movq	568(%rsp), %r14
	movzbl	576(%rsp), %r15d
	movzbl	536(%rsp), %r13d
	testq	%rbp, 648(%rsp)
	je	.LBB536_12
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_12
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_12:
	testb	$1, %r13b
	movabsq	$2305843009213693952, %r13
	je	.LBB536_18
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r12
	je	.LBB536_16
	lock		decq	-8(%rbx)
	jne	.LBB536_16
	addq	$-8, %rbx
	#MEMBARRIER
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_16:
	movl	$1, %ebx
	cmpb	$1, %r15b
	jne	.LBB536_82
.LBB536_17:
	movq	%r14, %rdi
	jmp	.LBB536_81
.LBB536_18:
	movq	40(%rsp), %rbx
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_retain@PLT
	xorl	%edi, %edi
	movq	$0, 40(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_TotalWritableBytes>>, struct<(scalar<index>) memoryOnly>]"@PLT
	cmpq	$23, %rax
	jg	.LBB536_22
	leaq	208(%rsp), %rdi
	movabsq	$-9223372036854775808, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%xmm0, 208(%rsp)
	movq	%rax, 224(%rsp)
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	movabsq	$4611686018427387904, %r15
	movq	224(%rsp), %rax
	testq	%r13, %rax
	jne	.LBB536_20
.LBB536_27:
	movl	$1336, %r14d
	testq	%rax, %rax
	js	.LBB536_29
	movq	216(%rsp), %rsi
	jmp	.LBB536_30
.LBB536_22:
	movq	%rax, %r14
	addq	$7, %r14
	movabsq	$4611686018427387904, %r15
	movq	%r14, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	js	.LBB536_96
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	testq	%rax, %rax
	je	.LBB536_98
	sarq	$3, %r14
	movq	$1, (%rax)
	addq	$8, %rax
	movq	$0, 4728(%rsp)
	orq	%r15, %r14
	movq	%rax, 208(%rsp)
	movq	$0, 216(%rsp)
	leaq	208(%rsp), %rax
	movq	%r14, 224(%rsp)
	leaq	632(%rsp), %r14
	movq	%rax, 4736(%rsp)
	movq	%r14, %rdi
	callq	"std::sys::compile::_DebugLevel::write_to[::Writer & ::AnyType](::_DebugLevel,$0),writer.T`2x1=[typevalue<#kgen.instref<std::format::_utils::_FlushingWriteBuffer,origin._mlir_origin`={  },origin={  },W=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],capacity_bytes=4096>>, struct<(struct<(array<4096, scalar<ui8>>) memoryOnly>, scalar<index>, pointer<none>) memoryOnly>]"@PLT
	movq	4728(%rsp), %rdx
	cmpq	$4096, %rdx
	jle	.LBB536_26
	movq	4736(%rsp), %rdi
	leaq	632(%rsp), %r14
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	xorl	%edx, %edx
	movq	$0, 4728(%rsp)
.LBB536_26:
	movq	4736(%rsp), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmOrigin,::Origin[False, $0]](::String,::Span[False, $0, ::SIMD[DType.uint8, 1], $1, AddressSpace.GENERIC])"@PLT
	movq	224(%rsp), %rax
	testq	%r13, %rax
	je	.LBB536_27
.LBB536_20:
	testq	%rax, %rax
	js	.LBB536_34
.LBB536_21:
	movq	208(%rsp), %rax
	jmp	.LBB536_35
.LBB536_29:
	bextrq	%r14, %rax, %rsi
.LBB536_30:
	incq	%rsi
	leaq	208(%rsp), %rdi
	callq	"std::collections::string::string::String::unsafe_ptr_mut(::String,::SIMD[DType.int, 1]$)"@PLT
	movq	224(%rsp), %rcx
	testq	%rcx, %rcx
	js	.LBB536_32
	movq	216(%rsp), %rcx
	jmp	.LBB536_33
.LBB536_32:
	bextrq	%r14, %rcx, %rcx
.LBB536_33:
	movb	$0, (%rax,%rcx)
	movq	224(%rsp), %rax
	orq	%r13, %rax
	movq	%rax, 224(%rsp)
	testq	%rax, %rax
	jns	.LBB536_21
.LBB536_34:
	leaq	208(%rsp), %rax
.LBB536_35:
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_54f1157311b42df6(%rip), %rdx
	leaq	static_string_e4d9b1a40c8247ca(%rip), %rcx
	leaq	static_string_45e6677a9f25c122(%rip), %r8
	leaq	48(%rsp), %rdi
	movl	$3760, %r9d
	movq	%rbx, %rsi
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$-1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_loadFunction@PLT
	addq	$32, %rsp
	.cfi_adjust_cfa_offset -32
	testq	%r15, 224(%rsp)
	je	.LBB536_38
	movq	208(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_38
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r14, %rax
.LBB536_38:
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 632(%rsp)
	movq	$0, 640(%rsp)
	movq	%r13, 648(%rsp)
	testq	%rax, %rax
	je	.LBB536_44
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f6c401c515dc2a68(%rip), %r9
	leaq	496(%rsp), %rdi
	leaq	640(%rsp), %rdx
	movl	$170, %ecx
	movl	$17, %r8d
	movq	%rax, %rsi
	pushq	$49
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	%r15, %rax
	movq	496(%rsp), %r14
	movq	512(%rsp), %rbp
	movq	520(%rsp), %r12
	movzbl	528(%rsp), %r13d
	movzbl	488(%rsp), %r15d
	testq	%rax, 648(%rsp)
	je	.LBB536_42
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_42
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_42:
	testb	$1, %r15b
	je	.LBB536_46
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movabsq	$4611686018427387904, %r15
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %rbp
	jne	.LBB536_77
	jmp	.LBB536_79
.LBB536_44:
	movq	40(%rsp), %r12
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	je	.LBB536_47
.LBB536_45:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r12, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%r13, %r12
	movzbl	8(%rsp), %r13d
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %rbp
	jne	.LBB536_77
	jmp	.LBB536_79
.LBB536_46:
	movb	%r13b, 8(%rsp)
	movq	%r12, %r13
	movabsq	$4611686018427387904, %r15
	movq	40(%rsp), %r12
	callq	"max::gpu::host::device_context::DeviceFunction::dump_rep[::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()],::Variant[::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path, ::TypeList[::AnyType, ::Bool, ::Path, ::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()], def() capturing thin -> ::Path]()]](max::gpu::host::device_context::DeviceFunction[$0, $1, $2, $3, $4, $5, $6, $7, $8]),func_type=[typevalue<#kgen.instref<std::builtin::_stubs::__MLIRType,T=[(!kgen.pointer<typevalue<#kgen.instref<~A~Qstd::builtin::variadics::VariadicPack,elt_is_mutable=false,origin._mlir_origin`={  },origin={  },element_trait=[typevalue<#kgen.trait_ref<[~A~Qstd::traits::anytype::AnyType~Q]>>, type],Ts.values`2=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],is_owned=false,Ts={  }~Q>>> imm_mem) -> !kgen.none, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none]>>, (!kgen.pointer<struct<(pointer<pointer<none>>, pointer<pointer<none>>, pointer<pointer<none>>) isParamPack>> imm_mem) -> !kgen.none],declared_arg_types.values`=[[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=bf16,length=1>>, scalar<bf16>],origin={  },address_space=0>>, pointer<none>],[typevalue<#kgen.instref<std::memory::pointer::Pointer,mut=true,origin._mlir_origin`={  },T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=f32,length=1>>, scalar<f32>],origin={  },address_space=0>>, pointer<none>]],target._mlir_value`1=#kgen.target<triple = ~Qamdgcn-amd-amdhsa~Q, arch = ~Qgfx1201~Q, stdlib_plugin = ~Qhip~Q, data_layout = ~Qe-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9~Q, simd_bit_width = 128, index_bit_width = 64>,func=def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea[LITImmOrigin,::Origin[False, $0],def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None](::VariadicPack[False, $0, $1, ::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], False, ::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()])<intr::wmma_kernel(::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC],::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC])>,compile_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },link_options={ #interp.memref<{[(#interp.memory_handle<16, ~Q~Q string>, const_global, [], [])], []}, 0, 0>, 0 },_ptxas_info_verbose=false,dump_asm={ { {:scalar<bool> false, 0} } },dump_llvm={ { {:scalar<bool> false, 0} } },_dump_sass={ { {:scalar<bool> false, 0} } }"@PLT
	testb	$1, %al
	jne	.LBB536_45
.LBB536_47:
	leaq	600(%rsp), %rcx
	leaq	607(%rsp), %rax
	movq	%r12, 8(%rsp)
	leaq	632(%rsp), %r14
	movq	$1, 632(%rsp)
	movq	$0, 640(%rsp)
	testq	%rcx, %rcx
	movq	%r14, %rdi
	cmovnsq	%rcx, %rax
	setns	%dl
	sarq	$3, %rax
	movq	%rax, %rsi
	negq	%rsi
	shlq	$3, %rsi
	addq	%rcx, %rsi
	movq	%rbx, %rsi
	setne	%cl
	andb	%dl, %cl
	movzbl	%cl, %r12d
	addq	%rax, %r12
	shlq	$3, %r12
	leaq	16(%r12), %r13
	leaq	8(%r12), %r15
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	movq	640(%rsp), %rdx
	cmpq	$5, %rdx
	jne	.LBB536_52
	movq	632(%rsp), %rax
	leaq	static_string_561d9efdd15f277f(%rip), %rcx
	cmpq	%rcx, %rax
	je	.LBB536_54
	movq	%rdx, %rsi
	sarq	$63, %rsi
	andnq	%rdx, %rsi, %rdx
	xorl	%esi, %esi
	.p2align	4
.LBB536_50:
	cmpq	%rsi, %rdx
	je	.LBB536_54
	movzbl	(%rax,%rsi), %edi
	cmpb	(%rsi,%rcx), %dil
	leaq	1(%rsi), %rsi
	je	.LBB536_50
.LBB536_52:
	movq	176(%rsp), %rax
	movq	184(%rsp), %rdi
	movq	192(%rsp), %rsi
	movq	8(%rsp), %r14
	movq	%r12, 208(%rsp)
	movq	%r15, 216(%rsp)
	movq	%r13, 224(%rsp)
	movq	$0, 584(%rsp)
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%rax, (%r12)
	leaq	208(%rsp), %rax
	movq	%rdi, 8(%r12)
	movq	%rsi, 16(%r12)
	movq	%rbx, %rdi
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	movabsq	$2305843009213693952, %rbp
	testq	%rax, %rax
	je	.LBB536_84
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	632(%rsp), %r14
	movq	%rax, 80(%rsp)
	movq	%rdx, 88(%rsp)
	movq	$1, 632(%rsp)
	movq	%rcx, 96(%rsp)
	movq	$0, 640(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	632(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 152(%rsp)
	movq	$0, 168(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 112(%rsp)
	movq	$12, 136(%rsp)
	movq	$1, 24(%rsp)
	movq	$13, 48(%rsp)
	movq	$1, 640(%rsp)
	movq	%rcx, 104(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 120(%rsp)
	movq	%rcx, 128(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 144(%rsp)
	movq	%rcx, 16(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 32(%rsp)
	movq	%rcx, 40(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 56(%rsp)
	movq	%rcx, 632(%rsp)
	movq	%rbp, 648(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	136(%rsp), %r12
	leaq	static_string_496d70d5cdfac6c1(%rip), %r13
	leaq	88(%rsp), %r10
	leaq	24(%rsp), %r11
	leaq	160(%rsp), %r15
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	456(%rsp), %rdi
	leaq	112(%rsp), %r9
	movl	$136, %esi
	movl	$42, %edx
	movl	$25, %r8d
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	leaq	64(%rsp), %r10
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r15
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	$869
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movzbl	480(%rsp), %r13d
	movq	472(%rsp), %r12
	movq	464(%rsp), %rbp
	movq	448(%rsp), %r14
	movabsq	$4611686018427387904, %r15
	testq	%r15, 648(%rsp)
	jne	.LBB536_56
	jmp	.LBB536_58
.LBB536_54:
	movq	176(%rsp), %rax
	movq	184(%rsp), %rsi
	movq	192(%rsp), %rdi
	movq	%r12, 208(%rsp)
	movq	%r15, 216(%rsp)
	leaq	16(%rsp), %r15
	vxorps	%xmm0, %xmm0, %xmm0
	vmovups	%xmm0, 19(%rsp)
	movb	$0, 16(%rsp)
	movb	$0, 17(%rsp)
	vmovups	%zmm0, 720(%rsp)
	vmovups	%zmm0, 656(%rsp)
	movq	$8, 632(%rsp)
	movq	$8, 640(%rsp)
	movq	%r13, 224(%rsp)
	movq	$8, 648(%rsp)
	movb	$0, 18(%rsp)
	movq	$0, 592(%rsp)
	movl	$1, %edx
	movl	$1, %ecx
	movl	$1, %r8d
	movl	$32, %r9d
	movq	%rax, (%r12)
	movq	%rsi, 8(%r12)
	leaq	208(%rsp), %rsi
	movq	%rdi, 16(%r12)
	leaq	40(%rsp), %r12
	leaq	200(%rsp), %rax
	movq	%rbx, %rdi
	movq	%rsi, 40(%rsp)
	movq	%r14, 48(%rsp)
	movq	8(%rsp), %r14
	movq	%r15, 56(%rsp)
	movq	$8, 64(%rsp)
	movl	$0, 72(%rsp)
	movq	%r12, 200(%rsp)
	movq	%r14, %rsi
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$3
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$4
	.cfi_adjust_cfa_offset 8
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	pushq	$1
	.cfi_adjust_cfa_offset 8
	vzeroupper
	callq	AsyncRT_DeviceContext_enqueueFunctionDirect@PLT
	addq	$64, %rsp
	.cfi_adjust_cfa_offset -64
	testq	%rax, %rax
	je	.LBB536_83
	movq	%rax, %rdi
	callq	"max::gpu::host::device_context::_string_from_owned_charptr[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]])"@PLT
	leaq	632(%rsp), %r14
	movq	%rax, 80(%rsp)
	movq	%rdx, 88(%rsp)
	movq	$1, 632(%rsp)
	movq	%rcx, 96(%rsp)
	movq	$0, 640(%rsp)
	movq	%rbx, %rsi
	movq	%r14, %rdi
	callq	AsyncRT_DeviceContext_deviceApi@PLT
	vmovups	632(%rsp), %xmm0
	movq	%rbx, %rdi
	vmovups	%xmm0, 152(%rsp)
	movq	$0, 168(%rsp)
	callq	AsyncRT_DeviceContext_id@PLT
	leaq	static_string_aaccaef1399538c1(%rip), %rcx
	movq	$17, 112(%rsp)
	movq	$12, 136(%rsp)
	movq	$1, 24(%rsp)
	movabsq	$2305843009213693952, %rbp
	movq	$13, 48(%rsp)
	movq	$1, 640(%rsp)
	movq	%rcx, 104(%rsp)
	leaq	static_string_d8f96e4c39e045bb(%rip), %rcx
	movq	%rbp, 120(%rsp)
	movq	%rcx, 128(%rsp)
	leaq	static_string_fd5c39b3eb3d3242(%rip), %rcx
	movq	%rbp, 144(%rsp)
	movq	%rcx, 16(%rsp)
	leaq	static_string_fe95cf738c304de4(%rip), %rcx
	movq	%rbp, 32(%rsp)
	movq	%rcx, 40(%rsp)
	leaq	static_string_e7661a0566dadf97(%rip), %rcx
	movq	%rbp, 56(%rsp)
	movq	%rcx, 632(%rsp)
	movq	%rbp, 648(%rsp)
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_496d70d5cdfac6c1(%rip), %r13
	movq	%r15, %rbp
	leaq	88(%rsp), %r10
	leaq	160(%rsp), %r11
	leaq	136(%rsp), %r15
	leaq	static_string_f078bd8d2bcbf530(%rip), %rcx
	leaq	416(%rsp), %rdi
	leaq	112(%rsp), %r9
	movl	$136, %esi
	movl	$42, %edx
	movl	$25, %r8d
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	pushq	%r12
	.cfi_adjust_cfa_offset 8
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r15
	.cfi_adjust_cfa_offset 8
	pushq	$869
	.cfi_adjust_cfa_offset 8
	pushq	%r13
	.cfi_adjust_cfa_offset 8
	callq	"std::builtin::error::Error::__init__[KGENParamList[::Writable & ::AnyType],*::Writable & ::AnyType,LITImmOrigin,::Origin[False, $2]](*$0),Ts.values`2x=[[typevalue<#kgen.instref<std::reflection::location::SourceLocation>>, struct<(scalar<index>, scalar<index>, struct<(pointer<none>, scalar<index>)>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string_span::StringSpan,mut=false,origin._mlir_origin`={  },origin={  }>>, struct<(pointer<none>, scalar<index>)>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::simd::SIMD,dtype=si64,length=1>>, scalar<si64>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>],[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]]"@PLT
	addq	$80, %rsp
	.cfi_adjust_cfa_offset -80
	movzbl	440(%rsp), %r13d
	movq	432(%rsp), %r12
	movq	424(%rsp), %rbp
	movq	408(%rsp), %r14
	movabsq	$4611686018427387904, %r15
	testq	%r15, 648(%rsp)
	je	.LBB536_58
.LBB536_56:
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_58
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_58:
	testq	%r15, 56(%rsp)
	je	.LBB536_61
	movq	40(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_61
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_61:
	testq	%r15, 32(%rsp)
	je	.LBB536_64
	movq	16(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_64
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_64:
	testq	%r15, 144(%rsp)
	je	.LBB536_67
	movq	128(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_67
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_67:
	testq	%r15, 120(%rsp)
	je	.LBB536_70
	movq	104(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_70
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_70:
	testq	%r15, 168(%rsp)
	je	.LBB536_73
	movq	152(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_73
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_73:
	testq	%r15, 96(%rsp)
	je	.LBB536_76
	movq	80(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_76
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_76:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	8(%rsp), %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testq	%r15, %rbp
	je	.LBB536_79
.LBB536_77:
	lock		decq	-8(%r14)
	jne	.LBB536_79
	addq	$-8, %r14
	#MEMBARRIER
	movq	%r14, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_79:
	movl	$1, %ebx
	cmpb	$1, %r13b
	jne	.LBB536_82
	movq	%r12, %rdi
.LBB536_81:
	callq	"std::memory::arc_pointer::ArcPointer::__deinit__(::ArcPointer[$0]$),T=[typevalue<#kgen.instref<std::memory::owned_pointer::OwnedPointer,T=[typevalue<#kgen.instref<std::simd::SIMD,dtype=ui8,length=1>>, scalar<ui8>]>>, pointer<none>]"@PLT
.LBB536_82:
	movq	%rbx, %rax
	addq	$4744, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB536_83:
	.cfi_def_cfa_offset 4800
	movabsq	$2305843009213693952, %rbp
.LBB536_84:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	movq	%r14, %rdi
	callq	AsyncRT_DeviceFunction_release@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_synchronize@PLT
	leaq	static_string_2d06800538d394c2(%rip), %rcx
	movq	%rcx, 632(%rsp)
	movq	$0, 640(%rsp)
	movq	%rbp, 648(%rsp)
	testq	%rax, %rax
	je	.LBB536_94
	subq	$8, %rsp
	.cfi_adjust_cfa_offset 8
	leaq	static_string_f078bd8d2bcbf530(%rip), %r9
	leaq	368(%rsp), %rdi
	leaq	640(%rsp), %rdx
	movl	$137, %ecx
	movl	$24, %r8d
	movq	%rax, %rsi
	pushq	$25
	.cfi_adjust_cfa_offset 8
	callq	"max::gpu::host::device_context::_raise_checked_impl[LITImmOrigin,::Origin[False, $0]](::Optional[::CStringSpan[$0, $1]],::String,::SourceLocation)_REMOVED_ARG"@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	368(%rsp), %r15
	movq	384(%rsp), %r12
	movq	392(%rsp), %r14
	movzbl	400(%rsp), %ebp
	movzbl	360(%rsp), %r13d
	movabsq	$4611686018427387904, %rax
	testq	%rax, 648(%rsp)
	je	.LBB536_89
	movq	632(%rsp), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB536_89
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	jne	.LBB536_90
	jmp	.LBB536_88
.LBB536_89:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
	testb	$1, %r13b
	je	.LBB536_88
.LBB536_90:
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r12
	je	.LBB536_93
	lock		decq	-8(%r15)
	jne	.LBB536_93
	addq	$-8, %r15
	#MEMBARRIER
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB536_93:
	movl	$1, %ebx
	cmpb	$1, %bpl
	je	.LBB536_17
	jmp	.LBB536_82
.LBB536_94:
	movq	%rbx, %rdi
	callq	AsyncRT_DeviceContext_release@PLT
.LBB536_88:
	xorl	%ebx, %ebx
	jmp	.LBB536_82
.LBB536_96:
	leaq	static_string_722168dfb05a6ad2(%rip), %rax
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	632(%rsp), %r8
	movl	$783, %edi
	movq	$35, 640(%rsp)
	jmp	.LBB536_97
.LBB536_98:
	leaq	static_string_09e773a88105e290(%rip), %rax
	movq	$37, 640(%rsp)
	leaq	static_string_ede496b2d17dd639(%rip), %rdx
	leaq	632(%rsp), %r8
	movl	$659, %edi
.LBB536_97:
	movq	%rax, 632(%rsp)
	movq	%r13, 648(%rsp)
	movl	$14, %esi
	movl	$33, %ecx
	callq	"std::os::os::_abort_report[::StringSpan[False, ImmStaticOrigin, ::Origin[False, ImmStaticOrigin]()],::Writable & ::AnyType](::SourceLocation,$1),prefix={ #interp.memref<{[(#interp.memory_handle<16, ~QABORT:\\00~Q string>, const_global, [], [])], []}, 0, 0>, 6 },message.T`=[typevalue<#kgen.instref<std::collections::string::string::String>>, struct<(pointer<none>, scalar<index>, scalar<index>) memoryOnly>]"@PLT
	ud2
.Lfunc_end34:
	.size	gate_dot2, .Lfunc_end34-gate_dot2
	.size	.Lgate_dot2$local, .Lfunc_end34-gate_dot2
	.cfi_endproc

	.type	static_string_cff33790b37cb0fb,@object
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
static_string_cff33790b37cb0fb:
	.asciz	"def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], c: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|16dc1a059582f1ea"
	.size	static_string_cff33790b37cb0fb, 870

	.type	static_string_74d279525b544160,@object
	.p2align	4, 0x0
static_string_74d279525b544160:
	.asciz	"intr_wmma_kernel_Pointer_True6A6AoA6A6AcB_54616a893a8b3fe9"
	.size	static_string_74d279525b544160, 59

	.type	static_string_4fc62907bcfcc19a,@object
	.p2align	4, 0x0
static_string_4fc62907bcfcc19a:
	.asciz	"13fff53b7d1e479311fbb2e7c9fdf0a9"
	.size	static_string_4fc62907bcfcc19a, 33

	.type	static_string_bdc25b2a53dd7d0b,@object
	.p2align	4, 0x0
static_string_bdc25b2a53dd7d0b:
	.asciz	"\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000p\f\000\000\000\000\000\000N\000\000\000@\0008\000\b\000@\000\017\000\r\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\300\001\000\000\000\000\000\000\300\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\006\000\000\000\000\000\000@\006\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\007\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\027\000\000\000\000\000\000\200\003\000\000\000\000\000\000\200\003\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\200\n\000\000\000\000\000\000\200*\000\000\000\000\000\000\200*\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\005\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\200\n\000\000\000\000\000\000\200*\000\000\000\000\000\000\200*\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\200\n\000\000\000\000\000\000\200*\000\000\000\000\000\000\200*\000\000\000\000\000\000p\000\000\000\000\000\000\000\200\005\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\304\002\000\000\000\000\000\000\304\002\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000\257\002\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\217\245.args\223\204\256.address_space\247generic\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\247generic\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\247generic\247.offset\020\245.size\b\253.value_kind\255global_buffer\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\030\270.max_flat_workgroup_size\315\004\000\245.name\331:intr_wmma_kernel_Pointer_True6A6AoA6A6AcB_54616a893a8b3fe9\273.private_segment_fixed_size\000\253.sgpr_count\n\261.sgpr_spill_count\000\247.symbol\331=intr_wmma_kernel_Pointer_True6A6AoA6A6AcB_54616a893a8b3fe9.kd\263.uses_dynamic_stack\302\253.vgpr_count\023\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\331!amdgcn-amd-amdhsa-unknown-gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\027\000\000\000\000\000\000\234\001\000\000\000\000\000\000<\000\000\000\021\000\006\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000\001\000\000\000\000\002\000\220\001\000\000\000>\207\223\002\275\204\335\247\003\000\000\000\003\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000intr_wmma_kernel_Pointer_True6A6AoA6A6AcB_54616a893a8b3fe9\000intr_wmma_kernel_Pointer_True6A6AoA6A6AcB_54616a893a8b3fe9.kd\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\000\021\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\002\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000A\000\364\000\000\000\370\201\000\0022\217\000\0006\000 \000\364\020\000\000\370\022\001\207\277\377\002\0046\370\001\000\000\205\000\n0\204\002\0020\243\001\207\277\003\000V\326\002\t\001\004\201\004\0040\000\000X\326\001\001\376\003p\000\000\000\003\000\207\277\201\006\b0\202\006$0\000\000\307\277\003\000\205\277\006\000\b\356\b\000\000\000\004\000\000\000\006\000\b\356\t\000\000\000\004@\000\000\006\000\b\356\n\000\000\000\004\200\000\000\006\000\b\356\013\000\000\000\004\300\000\000\005\002\000\327\004\n\002\002\021\001\207\277\006| \325\005\000\t\000\001j\000\327\005\005\002\002\001\000\207\277\002| \325\200\f\252\001\201\000\n0|\300\005\356\f\000\000\000\001\000\000\000\004\000\300\277\006\300\b\356\b\000\000\000\004 \000\000\004\000\300\277\006\300\b\356\t\000\000\000\004`\000\000\004\000\300\277\006\300\b\356\n\000\000\000\004\240\000\000\004\000\300\277\006\300\b\356\013\000\000\000\005\000\000\000\200\002\002~1\001\207\277\202\000 >\000\000\300\277\000@A\314\f\021\002\032\bj\000\327\000 \002\002\235\377\210\277\003\000\207\277\t| \325\001\"\252\001\007\000\205\277\000\200\006\356\000\000\000\000\022\000\000\000\000\200\006\356\000\000\200\000\022@\000\000\000\200\006\356\000\000\000\001\022\200\000\000\000\200\006\356\000\000\200\001\022\300\000\000\000\200\006\356\000\000\000\002\022\000\001\000\000\200\006\356\000\000\200\002\022@\001\000\000\200\006\356\000\000\000\003\022\200\001\000|\200\006\356\000\000\200\003\b\000\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\310\004\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000T\005\000\000\000\000\000\000\n\000\000\000\000\000\000\000z\000\000\000\000\000\000\000\365\376\377o\000\000\000\000\020\005\000\000\000\000\000\000\004\000\000\000\000\000\000\0004\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: LLD 24.0.0\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000z\000\000\000\000\002\b\000\200*\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\027\000\000\000\000\000\000\234\001\000\000\000\000\000\000<\000\000\000\021\000\006\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000intr_wmma_kernel_Pointer_True6A6AoA6A6AcB_54616a893a8b3fe9\000intr_wmma_kernel_Pointer_True6A6AoA6A6AcB_54616a893a8b3fe9.kd\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\304\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\310\004\000\000\000\000\000\000\310\004\000\000\000\000\000\000H\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\020\005\000\000\000\000\000\000\020\005\000\000\000\000\000\000$\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\0004\005\000\000\000\000\000\0004\005\000\000\000\000\000\000 \000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000T\005\000\000\000\000\000\000T\005\000\000\000\000\000\000z\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\007\000\000\000\000\000\000\200\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\200*\000\000\000\000\000\000\200\n\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000\360*\000\000\000\000\000\000\360\n\000\000\000\000\000\000\020\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\n\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000b\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\n\000\000\000\000\000\000\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000k\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\b\013\000\000\000\000\000\000`\000\000\000\000\000\000\000\016\000\000\000\002\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000s\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000h\013\000\000\000\000\000\000\205\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000}\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\355\013\000\000\000\000\000\000\203\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	static_string_bdc25b2a53dd7d0b, 4145

	.type	static_string_822c3714e12be7aa,@object
	.p2align	4, 0x0
static_string_822c3714e12be7aa:
	.asciz	"intr_loadtr_kernel_Pointer_Tr6A6AoA6A6AcB_1e3b11142e65683f"
	.size	static_string_822c3714e12be7aa, 59

	.type	static_string_49fbd36d4f006cd8,@object
	.p2align	4, 0x0
static_string_49fbd36d4f006cd8:
	.asciz	"065e65088f38285865259ea6d2cc9ef2"
	.size	static_string_49fbd36d4f006cd8, 33

	.type	static_string_e64fd374c68b723d,@object
	.p2align	4, 0x0
static_string_e64fd374c68b723d:
	.asciz	"\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\360\t\000\000\000\000\000\000N\000\000\000@\0008\000\b\000@\000\017\000\r\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\300\001\000\000\000\000\000\000\300\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\006\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\000\b\000\000\000\000\000\000\000(\000\000\000\000\000\000\000(\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\000\b\000\000\000\000\000\000\000(\000\000\000\000\000\000\000(\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\000\b\000\000\000\000\000\000\000(\000\000\000\000\000\000\000(\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\204\002\000\000\000\000\000\000\204\002\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000m\002\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\217\245.args\222\204\256.address_space\247generic\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\247generic\247.offset\b\245.size\b\253.value_kind\255global_buffer\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\020\270.max_flat_workgroup_size\315\004\000\245.name\331:intr_loadtr_kernel_Pointer_Tr6A6AoA6A6AcB_1e3b11142e65683f\273.private_segment_fixed_size\000\253.sgpr_count\004\261.sgpr_spill_count\000\247.symbol\331=intr_loadtr_kernel_Pointer_Tr6A6AoA6A6AcB_1e3b11142e65683f.kd\263.uses_dynamic_stack\302\253.vgpr_count\005\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\331!amdgcn-amd-amdhsa-unknown-gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\026\000\000\000\000\000\0000\000\000\000\000\000\000\000<\000\000\000\021\000\006\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000\000\000\220\000\000@\001\000\001\000\000\000\026\320E\272\225>A\302\003\000\000\000\003\000\000\000\000\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000intr_loadtr_kernel_Pointer_Tr6A6AoA6A6AcB_1e3b11142e65683f\000intr_loadtr_kernel_Pointer_Tr6A6AoA6A6AcB_1e3b11142e65683f.kd\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000@\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\000\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000@\000\364\000\000\000\370\204\000\b0\000\000\307\277\000\300\025\356\000\000\000\000\004\000\000\000\000\000\300\277\002@\007\356\000\000\000\000\004\000\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\210\004\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\024\005\000\000\000\000\000\000\n\000\000\000\000\000\000\000z\000\000\000\000\000\000\000\365\376\377o\000\000\000\000\320\004\000\000\000\000\000\000\004\000\000\000\000\000\000\000\364\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: LLD 24.0.0\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000z\000\000\000\000\002\b\000\000(\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\026\000\000\000\000\000\0000\000\000\000\000\000\000\000<\000\000\000\021\000\006\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000intr_loadtr_kernel_Pointer_Tr6A6AoA6A6AcB_1e3b11142e65683f\000intr_loadtr_kernel_Pointer_Tr6A6AoA6A6AcB_1e3b11142e65683f.kd\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\204\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\210\004\000\000\000\000\000\000\210\004\000\000\000\000\000\000H\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\320\004\000\000\000\000\000\000\320\004\000\000\000\000\000\000$\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000\364\004\000\000\000\000\000\000\364\004\000\000\000\000\000\000 \000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\024\005\000\000\000\000\000\000\024\005\000\000\000\000\000\000z\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\300\005\000\000\000\000\000\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\000(\000\000\000\000\000\000\000\b\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p(\000\000\000\000\000\000p\b\000\000\000\000\000\000\220\007\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000b\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\b\000\000\000\000\000\000\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000k\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\210\b\000\000\000\000\000\000`\000\000\000\000\000\000\000\016\000\000\000\002\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000s\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\350\b\000\000\000\000\000\000\205\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000}\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000m\t\000\000\000\000\000\000\203\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	static_string_e64fd374c68b723d, 3505

	.type	static_string_fad8c1e7bc02fc5f,@object
	.p2align	4, 0x0
static_string_fad8c1e7bc02fc5f:
	.asciz	"intr_lds_kernel_Pointer_True6A6AoA6A6AcBsA_4e0b19d777c1222a"
	.size	static_string_fad8c1e7bc02fc5f, 60

	.type	static_string_503eabc945fed7e4,@object
	.p2align	4, 0x0
static_string_503eabc945fed7e4:
	.asciz	"b47514aa953df472e486095b99cdd317"
	.size	static_string_503eabc945fed7e4, 33

	.type	static_string_938a41beea320a7b,@object
	.p2align	4, 0x0
static_string_938a41beea320a7b:
	.asciz	"\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\n\000\000\000\000\000\000N\000\000\000@\0008\000\b\000@\000\017\000\r\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\300\001\000\000\000\000\000\000\300\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\006\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\000\t\000\000\000\000\000\000\000)\000\000\000\000\000\000\000)\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\007\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\000\t\000\000\000\000\000\000\000)\000\000\000\000\000\000\000)\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\000\t\000\000\000\000\000\000\000)\000\000\000\000\000\000\000)\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\007\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\210\002\000\000\000\000\000\000\210\002\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000q\002\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\217\245.args\222\204\256.address_space\247generic\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\247generic\247.offset\b\245.size\b\253.value_kind\255global_buffer\271.group_segment_fixed_size\315\005\000\266.kernarg_segment_align\b\265.kernarg_segment_size\020\270.max_flat_workgroup_size\315\004\000\245.name\331;intr_lds_kernel_Pointer_True6A6AoA6A6AcBsA_4e0b19d777c1222a\273.private_segment_fixed_size\000\253.sgpr_count\004\261.sgpr_spill_count\000\247.symbol\331>intr_lds_kernel_Pointer_True6A6AoA6A6AcBsA_4e0b19d777c1222a.kd\263.uses_dynamic_stack\302\253.vgpr_count\f\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\331!amdgcn-amd-amdhsa-unknown-gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\026\000\000\000\000\000\000(\001\000\000\000\000\000\000=\000\000\000\021\000\006\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000\000 \000\000\001\020\001\000\001\000\000\0000\324\321\203\rx\214\262\003\000\000\000\003\000\000\000\001\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000intr_lds_kernel_Pointer_True6A6AoA6A6AcBsA_4e0b19d777c1222a\000intr_lds_kernel_Pointer_True6A6AoA6A6AcBsA_4e0b19d777c1222a.kd\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\005\000\000\000\000\000\000\020\000\000\000\000\000\000\000@\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0000\000\000\000\001\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000@\000\364\000\000\000\370\205\000\0220\201\000\0242\000\000\307\277\001\000\205\277\000\300\005\356\001\000\000\000\t\000\000\000\000\300\005\356\005\000\000\000\t\020\000\000\377\024\026\022\340\377\377\377\250\024\024\026\222\000\207\277\013\000F\326\000\t-\004\n\000G\326\013\025\006\002\001\000\300\277\000\000|\333\n\001\000\000\000\000\300\277\020\000|\333\n\005\000\000\000\000\306\277\301N\200\276\201\000\0200\377\377\224\277|\300\n\356\000\000\004\000\000\000\000\000\000\000\230\332\b\000\000\000P\000\234\332\b\000\000\000\240\000\230\332\b\000\000\001\360\000\234\332\b\000\000\001@\001\230\332\b\000\000\002\220\001\234\332\b\000\000\002\340\001\230\332\b\000\000\0030\002\234\332\b\000\000\003\200\002\230\332\b\000\000\004\320\002\234\332\b\000\000\004 \003\230\332\b\000\000\005p\003\234\332\b\000\000\005\300\003\230\332\b\000\000\006\020\004\234\332\b\000\000\006`\004\230\332\b\000\000\007\260\004\234\332\b\000\000\007\b\000\306\277\002@\007\356\000\000\000\000\t\000\000\000\000\000\306\277\002@\007\356\000\000\000\002\t\020\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\210\004\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\024\005\000\000\000\000\000\000\n\000\000\000\000\000\000\000|\000\000\000\000\000\000\000\365\376\377o\000\000\000\000\320\004\000\000\000\000\000\000\004\000\000\000\000\000\000\000\364\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: LLD 24.0.0\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000|\000\000\000\000\002\b\000\000)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\026\000\000\000\000\000\000(\001\000\000\000\000\000\000=\000\000\000\021\000\006\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000intr_lds_kernel_Pointer_True6A6AoA6A6AcBsA_4e0b19d777c1222a\000intr_lds_kernel_Pointer_True6A6AoA6A6AcBsA_4e0b19d777c1222a.kd\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\210\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\210\004\000\000\000\000\000\000\210\004\000\000\000\000\000\000H\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\320\004\000\000\000\000\000\000\320\004\000\000\000\000\000\000$\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000\364\004\000\000\000\000\000\000\364\004\000\000\000\000\000\000 \000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\024\005\000\000\000\000\000\000\024\005\000\000\000\000\000\000|\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\300\005\000\000\000\000\000\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\000)\000\000\000\000\000\000\000\t\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p)\000\000\000\000\000\000p\t\000\000\000\000\000\000\220\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\t\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000b\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\t\000\000\000\000\000\000\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000k\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\210\t\000\000\000\000\000\000`\000\000\000\000\000\000\000\016\000\000\000\002\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000s\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\350\t\000\000\000\000\000\000\205\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000}\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000m\n\000\000\000\000\000\000\205\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	static_string_938a41beea320a7b, 3769

	.type	static_string_f1dde65067a8a2a0,@object
	.p2align	4, 0x0
static_string_f1dde65067a8a2a0:
	.asciz	"def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(src: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], dst: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|bf2ab9340e9e5f13"
	.size	static_string_f1dde65067a8a2a0, 625

	.type	static_string_9aef310a067be18e,@object
	.p2align	4, 0x0
static_string_9aef310a067be18e:
	.asciz	"intr_nt_kernel_Pointer_True6A6AoA6A6AcBsAgA_a0c367a2d7944d1c"
	.size	static_string_9aef310a067be18e, 61

	.type	static_string_fe5419fd1e6ae11d,@object
	.p2align	4, 0x0
static_string_fe5419fd1e6ae11d:
	.asciz	"9e74a81c009ac0c97bb6e2781fea6269"
	.size	static_string_fe5419fd1e6ae11d, 33

	.type	static_string_707259faf29623c9,@object
	.p2align	4, 0x0
static_string_707259faf29623c9:
	.asciz	"\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\t\000\000\000\000\000\000N\000\000\000@\0008\000\b\000@\000\017\000\r\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\300\001\000\000\000\000\000\000\300\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\006\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\000\b\000\000\000\000\000\000\000(\000\000\000\000\000\000\000(\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\000\b\000\000\000\000\000\000\000(\000\000\000\000\000\000\000(\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\000\b\000\000\000\000\000\000\000(\000\000\000\000\000\000\000(\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\b\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\210\002\000\000\000\000\000\000\210\002\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000q\002\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\217\245.args\222\204\256.address_space\247generic\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\247generic\247.offset\b\245.size\b\253.value_kind\255global_buffer\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\020\270.max_flat_workgroup_size\315\004\000\245.name\331<intr_nt_kernel_Pointer_True6A6AoA6A6AcBsAgA_a0c367a2d7944d1c\273.private_segment_fixed_size\000\253.sgpr_count\004\261.sgpr_spill_count\000\247.symbol\331?intr_nt_kernel_Pointer_True6A6AoA6A6AcBsAgA_a0c367a2d7944d1c.kd\263.uses_dynamic_stack\302\253.vgpr_count\005\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\331!amdgcn-amd-amdhsa-unknown-gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\026\000\000\000\000\000\0000\000\000\000\000\000\000\000>\000\000\000\021\000\006\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000@\002\000\b\000\020\000\000\001\000\000\000\310wrl\007!\342\260\003\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000intr_nt_kernel_Pointer_True6A6AoA6A6AcBsAgA_a0c367a2d7944d1c\000intr_nt_kernel_Pointer_True6A6AoA6A6AcBsAgA_a0c367a2d7944d1c.kd\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000@\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\000\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000@\000\364\000\000\000\370\204\000\b0\000\000\307\277\000\300\005\356\000\000\020\000\004\000\000\000\000\000\300\277\002@\007\356\000\000\000\000\004\000\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\210\004\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\024\005\000\000\000\000\000\000\n\000\000\000\000\000\000\000~\000\000\000\000\000\000\000\365\376\377o\000\000\000\000\320\004\000\000\000\000\000\000\004\000\000\000\000\000\000\000\364\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: LLD 24.0.0\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000~\000\000\000\000\002\b\000\000(\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\026\000\000\000\000\000\0000\000\000\000\000\000\000\000>\000\000\000\021\000\006\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000intr_nt_kernel_Pointer_True6A6AoA6A6AcBsAgA_a0c367a2d7944d1c\000intr_nt_kernel_Pointer_True6A6AoA6A6AcBsAgA_a0c367a2d7944d1c.kd\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\210\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\210\004\000\000\000\000\000\000\210\004\000\000\000\000\000\000H\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\320\004\000\000\000\000\000\000\320\004\000\000\000\000\000\000$\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000\364\004\000\000\000\000\000\000\364\004\000\000\000\000\000\000 \000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\024\005\000\000\000\000\000\000\024\005\000\000\000\000\000\000~\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\300\005\000\000\000\000\000\000\300\005\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\026\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\000(\000\000\000\000\000\000\000\b\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p(\000\000\000\000\000\000p\b\000\000\000\000\000\000\220\007\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000b\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\b\000\000\000\000\000\000\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000k\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\210\b\000\000\000\000\000\000`\000\000\000\000\000\000\000\016\000\000\000\002\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000s\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\350\b\000\000\000\000\000\000\205\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000}\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000m\t\000\000\000\000\000\000\207\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	static_string_707259faf29623c9, 3513

	.type	static_string_f078bd8d2bcbf530,@object
	.p2align	4, 0x0
static_string_f078bd8d2bcbf530:
	.asciz	"/work/mojo_gate/intr.mojo"
	.size	static_string_f078bd8d2bcbf530, 26

	.type	static_string_aaccaef1399538c1,@object
	.p2align	4, 0x0
static_string_aaccaef1399538c1:
	.asciz	" failed calling '"
	.size	static_string_aaccaef1399538c1, 18

	.type	static_string_d8f96e4c39e045bb,@object
	.p2align	4, 0x0
static_string_d8f96e4c39e045bb:
	.asciz	"' on device "
	.size	static_string_d8f96e4c39e045bb, 13

	.type	static_string_fe95cf738c304de4,@object
	.p2align	4, 0x0
static_string_fe95cf738c304de4:
	.asciz	" with error '"
	.size	static_string_fe95cf738c304de4, 14

	.type	static_string_e7661a0566dadf97,@object
	.p2align	4, 0x0
static_string_e7661a0566dadf97:
	.asciz	"'"
	.size	static_string_e7661a0566dadf97, 2

	.type	static_string_561d9efdd15f277f,@object
	.p2align	4, 0x0
static_string_561d9efdd15f277f:
	.asciz	"metal"
	.size	static_string_561d9efdd15f277f, 6

	.type	static_string_f6c401c515dc2a68,@object
	.p2align	4, 0x0
static_string_f6c401c515dc2a68:
	.asciz	"max/mojo/max/gpu/host/_device_context_extras.mojo"
	.size	static_string_f6c401c515dc2a68, 50

	.type	static_string_69b8ed51c73424d7,@object
	.p2align	4, 0x0
static_string_69b8ed51c73424d7:
	.asciz	"max/mojo/max/gpu/host/device_context.mojo"
	.size	static_string_69b8ed51c73424d7, 42

	.type	static_string_3b45d8d0deb8f5b3,@object
	.p2align	4, 0x0
static_string_3b45d8d0deb8f5b3:
	.asciz	"hip"
	.size	static_string_3b45d8d0deb8f5b3, 4

	.type	static_string_496d70d5cdfac6c1,@object
	.p2align	4, 0x0
static_string_496d70d5cdfac6c1:
	.asciz	"def[LITImmOrigin, ::Origin[False, $0]](*args: *::TypeList[::AnyType, ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]]()) thin -> None|def(a: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], b: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.bfloat16, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC], o: ::Pointer[True, MutUnsafeAnyOrigin, ::SIMD[DType.float32, 1], ::Origin[True, MutUnsafeAnyOrigin](), AddressSpace.GENERIC]) thin -> None|7d11e2c78e2cb1d7"
	.size	static_string_496d70d5cdfac6c1, 870

	.type	static_string_e4d9b1a40c8247ca,@object
	.p2align	4, 0x0
static_string_e4d9b1a40c8247ca:
	.asciz	"intr_dot2_kernel_Pointer_True6A6AoA6A6AcB_0c893217b1638ad6"
	.size	static_string_e4d9b1a40c8247ca, 59

	.type	static_string_54f1157311b42df6,@object
	.p2align	4, 0x0
static_string_54f1157311b42df6:
	.asciz	"97385c952afbcccdd53302e5c53fa2d9"
	.size	static_string_54f1157311b42df6, 33

	.type	static_string_45e6677a9f25c122,@object
	.p2align	4, 0x0
static_string_45e6677a9f25c122:
	.asciz	"\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\360\n\000\000\000\000\000\000N\000\000\000@\0008\000\b\000@\000\017\000\r\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\300\001\000\000\000\000\000\000\300\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\006\000\000\000\000\000\000@\006\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\007\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\000\t\000\000\000\000\000\000\000)\000\000\000\000\000\000\000)\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\007\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\000\t\000\000\000\000\000\000\000)\000\000\000\000\000\000\000)\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\000\t\000\000\000\000\000\000\000)\000\000\000\000\000\000\000)\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\007\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\304\002\000\000\000\000\000\000\304\002\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\000\257\002\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\221\217\245.args\223\204\256.address_space\247generic\247.offset\000\245.size\b\253.value_kind\255global_buffer\204\256.address_space\247generic\247.offset\b\245.size\b\253.value_kind\255global_buffer\204\256.address_space\247generic\247.offset\020\245.size\b\253.value_kind\255global_buffer\271.group_segment_fixed_size\000\266.kernarg_segment_align\b\265.kernarg_segment_size\030\270.max_flat_workgroup_size\315\004\000\245.name\331:intr_dot2_kernel_Pointer_True6A6AoA6A6AcB_0c893217b1638ad6\273.private_segment_fixed_size\000\253.sgpr_count\b\261.sgpr_spill_count\000\247.symbol\331=intr_dot2_kernel_Pointer_True6A6AoA6A6AcB_0c893217b1638ad6.kd\263.uses_dynamic_stack\302\253.vgpr_count\003\261.vgpr_spill_count\000\257.wavefront_size \271.workgroup_processor_mode\001\255amdhsa.target\331!amdgcn-amd-amdhsa-unknown-gfx1201\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\027\000\000\000\000\000\000P\000\000\000\000\000\000\000<\000\000\000\021\000\006\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000\032\000\000\000\000\000\020\001@\002\000\000\001\000\000\000((\360Qgu\366a\003\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000intr_dot2_kernel_Pointer_True6A6AoA6A6AcB_0c893217b1638ad6\000intr_dot2_kernel_Pointer_True6A6AoA6A6AcB_0c893217b1638ad6.kd\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\000\021\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\000\000\017\340\204\000\000\000\b\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000A\000\364\000\000\000\370\202\000\0000\000 \000\364\020\000\000\370\000\000\307\277\001\000\205\277\004\000\005\356\001\000\000\000\000\000\000\000\006\000\005\356\002\000\000\000\000\000\000\000\000\000\300\277\001@\032\314\001\005\302\033\000\200\006\356\000\000\200\000\000\000\000\000\000\000\260\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\000\000\237\277\006\000\000\000\000\000\000\000\310\004\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000T\005\000\000\000\000\000\000\n\000\000\000\000\000\000\000z\000\000\000\000\000\000\000\365\376\377o\000\000\000\000\020\005\000\000\000\000\000\000\004\000\000\000\000\000\000\0004\005\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Linker: LLD 24.0.0\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000z\000\000\000\000\002\b\000\000)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000\027\000\000\000\000\000\000P\000\000\000\000\000\000\000<\000\000\000\021\000\006\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000intr_dot2_kernel_Pointer_True6A6AoA6A6AcB_0c893217b1638ad6\000intr_dot2_kernel_Pointer_True6A6AoA6A6AcB_0c893217b1638ad6.kd\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\002\000\000\000\000\000\000\304\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\310\004\000\000\000\000\000\000\310\004\000\000\000\000\000\000H\000\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000\020\005\000\000\000\000\000\000\020\005\000\000\000\000\000\000$\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\0004\005\000\000\000\000\000\0004\005\000\000\000\000\000\000 \000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\000T\005\000\000\000\000\000\000T\005\000\000\000\000\000\000z\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\006\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000\027\000\000\000\000\000\000\000\007\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\000)\000\000\000\000\000\000\000\t\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p)\000\000\000\000\000\000p\t\000\000\000\000\000\000\220\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\t\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000b\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000p\t\000\000\000\000\000\000\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000k\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\210\t\000\000\000\000\000\000`\000\000\000\000\000\000\000\016\000\000\000\002\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000s\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\350\t\000\000\000\000\000\000\205\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000}\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000m\n\000\000\000\000\000\000\203\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	static_string_45e6677a9f25c122, 3761

	.type	global_constant,@object
global_constant:
	.asciz	"At \000: \000"
	.size	global_constant, 8

	.type	static_string_fd5c39b3eb3d3242,@object
	.p2align	4, 0x0
static_string_fd5c39b3eb3d3242:
	.asciz	":"
	.size	static_string_fd5c39b3eb3d3242, 2

	.type	static_string_2d06800538d394c2,@object
	.p2align	4, 0x0
static_string_2d06800538d394c2:
	.size	static_string_2d06800538d394c2, 0

	.type	static_string_978d8d34847e5196,@object
	.p2align	4, 0x0
static_string_978d8d34847e5196:
	.asciz	"0123456789abcdefghijklmnopqrstuvwxyz"
	.size	static_string_978d8d34847e5196, 37

	.type	static_string_ba261bf194cae289,@object
	.p2align	4, 0x0
static_string_ba261bf194cae289:
	.asciz	"none"
	.size	static_string_ba261bf194cae289, 5

	.type	static_string_0d78baac08237ddb,@object
	.p2align	4, 0x0
static_string_0d78baac08237ddb:
	.asciz	"a"
	.size	static_string_0d78baac08237ddb, 2

	.type	static_string_31203c1a2bdb78cc,@object
	.p2align	4, 0x0
static_string_31203c1a2bdb78cc:
	.asciz	"ABORT:"
	.size	static_string_31203c1a2bdb78cc, 7

	.type	static_string_7f1562353e292282,@object
	.p2align	4, 0x0
static_string_7f1562353e292282:
	.asciz	": "
	.size	static_string_7f1562353e292282, 3

	.type	static_string_bbe01a6a523daf15,@object
	.p2align	4, 0x0
static_string_bbe01a6a523daf15:
	.asciz	"\n"
	.size	static_string_bbe01a6a523daf15, 2

	.type	static_string_ede496b2d17dd639,@object
	.p2align	4, 0x0
static_string_ede496b2d17dd639:
	.asciz	"Mojo/stdlib/std/memory/alloc.mojo"
	.size	static_string_ede496b2d17dd639, 34

	.type	static_string_09e773a88105e290,@object
	.p2align	4, 0x0
static_string_09e773a88105e290:
	.asciz	"alloc failed: returned a null pointer"
	.size	static_string_09e773a88105e290, 38

	.type	static_string_722168dfb05a6ad2,@object
	.p2align	4, 0x0
static_string_722168dfb05a6ad2:
	.asciz	"alloc: `Layout.count()` must be > 0"
	.size	static_string_722168dfb05a6ad2, 36

	.type	static_string_c44bdff4074eecdb,@object
	.p2align	4, 0x0
static_string_c44bdff4074eecdb:
	.zero	1
	.size	static_string_c44bdff4074eecdb, 1

	.type	static_string_a8d4ace0dc8d360e,@object
	.p2align	4, 0x0
static_string_a8d4ace0dc8d360e:
	.asciz	" "
	.size	static_string_a8d4ace0dc8d360e, 2

	.section	".note.GNU-stack","",@progbits
