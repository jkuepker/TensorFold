# Mojo Phase 0 toolchain gate (R9700, gfx1201)

Question: can Mojo (MAX 26.6 / Mojo 1.1.0) build and run GPU kernels on the Radeon AI PRO R9700
(gfx1201, RDNA4, wave32) and be called from a ROCm-PyTorch process? No kernels were ported.

Run 2026-10-04 on tatooine card B (PCI 07:00.0), fleetq job #454. Container `mojo-gate` from
`stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25`
(torch 2.11.0+rocm7.14, Python 3.12.3), flags `--device /dev/kfd --device /dev/dri --group-add 993
--group-add 44 --ipc=host --shm-size=16g --network host -e GPU_MAX_HW_QUEUES=1 -e ROCR_VISIBLE_DEVICES=1`.
Torch inside saw 1 device, `pci_bus_id=7`, `gcnArchName=gfx1201`.

## Setup (see setup.sh)

Stable index worked, nightly not needed:

    uv venv --system-site-packages --python /opt/vllm/bin/python /work/venv
    uv pip install --python /work/venv/bin/python "max[all]" mojo --extra-index-url https://whl.modular.com/simple/
    echo /opt/vllm/lib/python3.12/site-packages > /work/venv/lib/python3.12/site-packages/zz_torch.pth

Result: max 26.6.0, mojo 1.1.0 in a separate venv; the image's torch is untouched and reached through
the .pth. Gotchas:
- Mojo/MAX dlopen `libamdhip64.so`, which is only in `/opt/rocm/core-7.14/lib`; set
  `LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib` (see run.sh) or MAX reports "Failed to open library".
- In 1.1.0 the GPU stdlib moved under `max.*`: `from max.gpu import global_idx, barrier`,
  `from max.gpu.host import DeviceContext`, `from extensibility import InputTensor, OutputTensor, register`.
  `std.gpu` does not exist. `Int` kernel args are rejected (use Int32/Int64).
- `max.torch` is now `max.experimental.torch`; a custom-op library must be a directory with `__init__.mojo`.

## Results

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Toolchain | PASS | `Mojo 1.1.0 (8189361e)`; `import max, torch` -> `torch.cuda.is_available()=True`, 1 device; `driver.Accelerator()` api=hip, gfx1201 |
| 2 | vector add via DeviceContext | PASS | 1M floats, 0 mismatches (vecadd.mojo) |
| 3A | max.experimental.torch.CustomOpLibrary | PASS | bf16 `2*x+y` op exact vs torch, also on a non-default torch stream; 42 us/call vs 1.4 us torch.add (median of 1000, async). Op call does not block the host behind queued torch work (0.3 ms call, work still pending). Inputs are handed over with torch's current stream via dlpack; the op runs on MAX's session stream with ordering preserved in tests |
| 3B | `--emit shared-lib` + ctypes | PASS | exact result on Mojo's own stream and on torch's current/non-default `hipStream_t` via `DeviceContext.create_external_stream` (async, ordered). Overhead 1.2 ms/call as written because the DeviceContext and compile are rebuilt per call: no way found to keep one alive across C calls (`Pointer.write` needs trivial-del types, no `steal_data` on OwnedPointer/List, no `__disable_del`). Needs a different handle scheme before Phase 1 relies on B |
| 4a | bf16 WMMA 16x16x16 wave32 | PASS | `llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16` via `llvm_intrinsic` (bf16 vectors rejected, bitcast to int16); max abs err 2.4e-7 vs torch; ISA `v_wmma_f32_16x16x16_bf16` |
| 4b | global.load.tr.b128 | PASS | `llvm.amdgcn.global.load.tr.b128.v8bf16`, pointer cast to GLOBAL address space; result is exactly the per-8-lane 8x8 transpose of 16-bit elements; ISA `global_load_tr_b128` |
| 4c | LDS tile with row padding | PASS | `stack_allocation[..., address_space=SHARED]`, stride +16 B, transposed read equals torch; ISA `ds_store_b128` |
| 4d | nontemporal loads | PASS | `ptr.load[width=8, non_temporal=True]`; ISA `th:TH_LOAD_NT`; data exact |
| 4e | fdot2 | PASS | `llvm.amdgcn.fdot2.f32.bf16`; exact vs torch; ISA `v_dot2_f32_bf16` |
| 5 | bandwidth, 1 GiB, 16 B loads | PASS | Mojo 621.6 GB/s (grid 1024 x 256, grid-stride, reduction verified) vs torch bf16 `sum(dtype=f32)` 548 GB/s and torch int32 `sum()` 119 GB/s; card peak ~640 |

ISA dumps (`mojo build --emit asm --target-accelerator gfx1201`) are in `isa/*.amdgcn`.

## Recommendation for Phase 1

Path A (`max.experimental.torch.CustomOpLibrary`): it works end to end on ROCm torch, dispatches
asynchronously and is correct on non-default streams, at ~40 us/call overhead (fine for large
GEMM-like kernels, not for tiny ones). Path B gives direct control of the torch stream and near-zero
marshalling, but the per-call DeviceContext must be solved first. Both reach every intrinsic the
qmm_rocm.cu wmma kernel uses.

## Not verified

- Whether MAX's op stream is torch's stream (it is not the same handle; ordering held in tests, no formal proof).
- Path B overhead with a persistent context.
- The cuda-source kernel itself (not ported); nontemporal and tr loads were only tested on 16-byte-per-lane tiles.

## Files

setup.sh, run.sh, run_all.sh, vecadd.mojo, ops/ (path A op), pathA.py, pathB.mojo/.py, intr.mojo/.py,
isa_dump.sh, bw.mojo/.py, isa/.

## Phase 1: the lane matmul in Mojo (`TF_ROCM_LANE=mojo`)

`src/tensorfold/cuda/kernels/qmm_rocm.mojo` ports `qmm_rocm.cu`'s `wmma_kernel` (one tile: `wmma_mt1`, two tiles:
`wmma_mt2`, the only instantiations the wmma lane launches) and `reduce_kernel` (the dot2 lane's; the wmma lane adds
its K slices in-kernel, so the mojo lane never launches it: it is ported and tested on its own). `cuda/mojo.py`
builds the source to .hsaco at first use (cached under the torch extensions dir by source hash + `mojo --version` +
arch); `qmm_rocm_mojo.cu` loads them once and launches with hipModuleLaunchKernel on torch's current stream, with
`gemv_groups`' schedule (32-row passes, shape-fixed K slices, persistent grid from module occupancy).

Bits: equal to the HIP wmma lane on every test_qmm_rocm.py shape at rows 1, 4, 8, 12, 16, 17, 33, 40 and on the 27B's
projections at rows 1, 4, 8, 12, 16, bf16 and fp32 out (tests/cuda/test_qmm_rocm_mojo.py).

Speed (qmm_bench.py, results in qmm_bench_results.txt, cold weights, graph-replayed): at rows 1-16 the Mojo lane
takes 69-99% of the HIP lane's time on every 27B projection (worst: attn q at 16 rows, 0.99). What it took to get there from a literal port (10-50%
slower at first):
- 32-bit unsigned index math (Mojo `Int` is 64-bit signed: floor div/mod sequences and 64-bit address pairs);
- `pair()` written per word: as a 4-wide vector LLVM narrowed it to 16-bit ops (240 v_and/or_b16 vs 64 v_and_or_b32);
- scales and biases in 32-bit lanes: packed two to a VGPR, the d16_hi load waits out the d16 one, a full memory
  round trip inside the slab-ahead prefetch (hipcc packs some too: 4 d16_hi loads; with the look-ahead below, this
  is why the small projections now run 15-30% faster than the HIP lane);
- `llvm.amdgcn.sched.barrier` per K step: the scheduler hoisted all 16 A reads (64 VGPRs), 195 VGPRs cost the
  second block per CU (hipcc: 153; Mojo now 179, both 2 blocks/CU);
- the next slab's input rows and group sums loaded a step ahead, before that step's weight prefetch (loads return in
  order, so staging no longer waits on DRAM latency);
- 32-bit output/partial indices (64-bit ones were hoisted out of the loop and spilled the two-tile kernel), after
  which the two-tile kernel keeps the same one barrier per K step (251 VGPRs, no spills).
The two-tile kernel (rows 17-32, not in the acceptance rows) is 66-108% of HIP's time (gate/up, gu and down at
24 rows 3-8% slower; run to run they vary by a few percent).
Absolute GB/s reach ~670, above the card's nominal ~640: some MALL reuse between rotated copies is likely, so
compare the lanes by ratio.

## Phase 3: the decode tree attention in Mojo (`TF_ROCM_TREE_KERNEL=mojo`)

`src/tensorfold/cuda/kernels/attention_rocm.mojo` ports `attention_rocm.cu`'s `shared_kernel` (all four PIPE x KV8
instantiations: `shared_pipe`, `shared_pipe8`, `shared_flat`, `shared_flat8`), `tail_kernel` (`tail16`, `tail8`) and
attention.py's Triton `_merge` (`merge`). `attention_rocm_mojo.cu` loads the code objects (built at first use by
`cuda/mojo.py`, cached by source hash) and launches them with hipModuleLaunchKernel on torch's current stream, with
`tree_shared`'s and `tree_tail`'s grids and `rocm_launch`'s compute waves and schedule; `attention.py` routes
`TF_ROCM_TREE_KERNEL=mojo` there (the default is unchanged). `_paths` and prompt attention stay as they are.

Bits: the partials (o, m, l) of the shared and tail kernels and the merged rows equal the HIP kernels' (and Triton's
merge) on every head-size-256 case of test_attention.py (trees and chains, 1, 2, 5, 8 compute waves with and without
loader waves, several streams in one launch, padded plans, strided node values), and on the 27B's geometry (24 query
heads, 4 KV heads, 256) at 1,013, 16,371 and 64,007 committed keys with 1-16 verify rows, bf16 and packed FP8 caches
(tests/cuda/test_attention_rocm_mojo.py). What it took (from the ISA of hipcc's and Triton's code objects):
- Mojo contracts `a * b + c` into an fma by default (hipcc too, Triton at the backend), so every step is written
  out: the fold's `l = fma(l, alpha, sum)`, the merge's `o = fma(o, a, b * co)` and `l = fma(cl, b, l * a)`;
- HIP's `__expf` is `v_exp_f32(x * log2e)` (`llvm.amdgcn.exp2`), Triton's `tl.exp` is `llvm.exp2`'s range-reduced
  lowering with the multiply fused into the scaling add: `ldexp(v_exp(fma(x, log2e, x*log2e < -126 ? 64 : 0)), ...)`;
- the merge's division is precise (`v_div_scale/fmas/fixup` in both), bf16 nearest even with NaN as 0x7FFF;
- running max as `llvm.maxnum` (`v_max_num`), the tile layout and WMMA operand order as the HIP kernel's.

Speed (attn_bench.py, attn_bench_results.txt; cold KV, graph-replayed, one attention call = shared + tail + merge):
bf16 caches 0.67-0.92 of HIP's time (worst 128k at 12 rows), packed FP8 0.68-0.96; the merge 0.55-0.97, the tail
1.00-1.08 (+0.3 us). What it took from a literal port (shared kernel 1.7x slower at 12 rows, every kernel spilling):
- loads from a wave-uniform base at a 32-bit byte offset: `global_load_b128 v, voff, s[base]`, no 64-bit address
  math a piece (hipcc's has a v_mad_co_u64_u32 per piece);
- the LDS operand base per lane hoisted, so the 16 K and 16 V reads a tile take immediate offsets (an address
  computed as `(base + 256 n) >> 1` kept 16 address registers, which spilled the query);
- no SIMD vector merged across branches: a conditional insert into a 16-wide register vector became a phi that LLVM
  shuffled with dozens of moves and `s_wait_loadcnt` between loads (each load waited on the last). The flat stage
  branches once on its wave's piece count (a wave-uniform count via readfirstlane: pieces and blocks are multiples
  of 32) and loads and places in the same branch; the packed stage, where that form spilled, guards per piece;
- V transposed by the load (`global_load_tr_b128`): 8 keys of one dim land in a lane, one `ds_store_b128` a piece
  instead of eight `ds_store_b16` (bf16 only; ~1%);
- the tail's piece loads branch-free (one address select, one load, a zero select), and a fifth wave folds while
  four loader waves fill a double buffer (hipcc's tail spills; the Mojo one had 84 bytes, now 28).
Tried and dropped: four tiles in loader registers instead of two (no gain at 12 rows, the FP8 pipe kernel spilled),
nontemporal key/value loads (2-5% slower). At 12 rows the shared kernel reaches ~470-490 GB/s against ~600 at one
row: there the five compute waves' WMMA chains and softmax, not the loads, set the pace.

Tests: test_attention_rocm_mojo.py 43 passed; with `TF_ROCM_LANE=mojo TF_ROCM_TREE_KERNEL=mojo` the attention, draft,
GLM draft ring and Qwen27 GPU tests plus test_cuda_build/test_cuda_kv8 give 399 passed, 4 failed, 3 errors, the same
as with `TF_ROCM_TREE_KERNEL=wmma` (test_glm_draft_ring's flat-buffer bits and the GLM experts extension not
building in this image).
