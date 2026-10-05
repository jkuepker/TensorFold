# Mojo Phase 4 toolchain gate (DGX Spark / GB10, sm_121, aarch64)

Question: can Mojo (MAX 26.6.0 / Mojo 1.1.0) build and run GPU kernels on the GB10 (Blackwell sm_121, 48 SMs,
unified LPDDR5x ~273 GB/s) on Linux aarch64, and be launched from an NGC PyTorch process the way Phases 0 and 1a
did on the R9700? No TensorFold kernels were ported.

Run 2026-10-04 on spark2 (spark-fa00), fleetq job #475 (8 GB; its vLLM `retro` role stayed up). Container `mojo-p4`
from `nvcr.io/nvidia/pytorch:26.07-py3` (torch 2.13.0a0+9186a08b2c.nv26.07, CUDA 13.3.73, driver 580.173.02,
Python 3.12.3), flags `--gpus all --ipc=host --network host -v ~/mojo-p4:/work`.

## Setup (setup.sh)

    uv venv --system-site-packages --python /usr/bin/python3 /work/venv
    uv pip install --python /work/venv/bin/python "max[all]" mojo --extra-index-url https://whl.modular.com/simple/

aarch64 wheels exist on the stable index: max 26.6.0, max-core 26.6.0, mojo 1.1.0, mojo-compiler 1.1.0
(`mojo --version` = `Mojo 1.1.0 (8189361e)`). NGC's torch is untouched (the venv sees it through
`--system-site-packages`; uv installed no torch), `import max, torch` leaves `torch.cuda.is_available()` True.
No LD_LIBRARY_PATH needed (unlike ROCm: libcuda is found). Mojo reports the device as `NVIDIA GB10`, api `cuda`,
`driver.Accelerator()` architecture `sm_121a`.
Import paths in 1.1.0: `max.gpu.compute.mma` (`mma`, `ld_matrix`, `st_matrix`), `max.gpu.memory` (`async_copy*`,
`cp_async_bulk_tensor_*`), `max.gpu.host.DeviceContext`; `std.sys.inlined_assembly`.

## Results

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Install | PASS | aarch64 wheels for max/mojo 26.6.0/1.1.0 on whl.modular.com stable; torch.cuda still works after `import max` |
| 2 | vector add via DeviceContext | PASS | 1M floats, 0 mismatches; Mojo reports `NVIDIA GB10`, api cuda (vecadd.mojo) |
| 3 | launch path | PASS | see below; exact bf16 `2*x+y`, non-default stream, CUDA graph of 10 launches, 2.8 us/call = nvcc `<<<>>>` |
| 4a | `mma.sync.m16n8k16` bf16 | PASS | `gpu.mma.mma()` with SIMD[bf16,8]/[bf16,4]/[f32,4]; max abs err 9.5e-7 vs torch; SASS `HMMA.16816.F32.BF16` |
| 4b | `ldmatrix` | PASS | `ld_matrix[8](p)` (x4) and `ld_matrix[4, transpose=True](p)` (x2.trans); fed the mma above, same error; SASS `LDSM.16.M88.4`, `LDSM.16.MT88.2` |
| 4c | `cp.async` | PASS | `async_copy[16](global_ptr, shared_ptr)` + commit_group + wait_all; PTX `cp.async.cg.shared.global`, SASS `LDGSTS.E.BYPASS.128` + `LDGDEPBAR`/`DEPBAR`. TMA: `cp_async_bulk_tensor_shared_cluster_global` is importable but NOT exercised (needs a host-built CUtensorMap) |
| 4d | shared tile with row padding | PASS | `stack_allocation[..., address_space=SHARED]`, A stride 24, B stride 16 bf16 (+16 B per row), cp.async-filled and ldmatrix-read, result exact |
| 4e | fp8 e4m3 `mma.sync.m16n8k32` | PASS | `mma()` accepts `float8_e4m3fn` (SIMD[.,16], SIMD[.,8]); max abs err 0 vs torch; SASS `QMMA.16832.F32.E4M3.E4M3` |
| 4e | block-scaled fp4 `mma.sync.m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::4X` e2m1 x ue4m3 | PASS (instruction + scales; fragment layout not verified) | inline PTX (no `mma()` wrapper); builds and runs, SASS `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`; all-ones operands with scales (1,2,4,8) give 240, doubling B's scales gives 480, unit scales 64. Per-lane fragment layout was not checked against a reference |
| 5 | bandwidth, 1 GiB, 16 B loads | PASS | Mojo 233.6 GB/s best (grid 3072 x 256; 230-234 flat across 192..12288 blocks; 100-200 reps give 233-233.4), reduction matches torch; torch bf16 `sum(dtype=f32)` 191 GB/s at 20 reps / 237.7 GB/s at 100 reps (warm), torch int32 `sum()` 43 GB/s. ~85% of the 273 GB/s peak |

## Check 3: the launch path (cumodule/)

`mojo build --emit object --target-accelerator sm_121` works; `sm_121`, `sm_121a` give identical objects (target
`sm_121a`, PTX ISA `.version 8.8`), `sm_120` gives `sm_120a` PTX (different bytes; not run), `DGXSpark` is rejected ("GPU architecture 'DGXSpark' is not supported"; the accepted
list includes `sm_121 (DGX Spark)`). Unlike AMD, the host object embeds **PTX text** per kernel (one module per
instantiated kernel, NUL terminated), not a finished device ELF. `cumodule/mojo2cubin.py` carves each PTX out, parses
the `.entry` `.param` list into the manifest (index, type, offset, size, pointer flag; no hidden arguments on NVIDIA),
and runs `ptxas -arch=sm_121a` (CUDA 13.3's) to make a cubin per kernel. Both load as is:
`cuModuleLoadData(cubin)` 0.1-0.2 ms; `cuModuleLoadData(ptx)` JITs in the driver (62 ms the first time, then served
from the driver's JIT cache). `cuobjdump -sass` on the cubins works.

`cumodule/bench.py` builds a torch extension (`load_inline`, nvcc, `-lcuda`, `TORCH_CUDA_ARCH_LIST=12.1`) that loads a
module and calls `cuLaunchKernel` on `c10::cuda::getCurrentCUDAStream()`:
- bf16 `2*x+y` exact vs torch for n = 1, 255, 256, 257, 4096, 1M, from both the cubin and the JITted PTX;
- non-default stream: consumed the output of five 8192^2 matmuls queued on that stream, exact;
- a kernarg-probe kernel (Int32/Int64/pointer mix) receives 11, 22, 33, 44 through the packed-buffer launch form built
  from the manifest offsets, so the manifest layout is right;
- `torch.cuda.CUDAGraph` capture of 10 chained launches, replay, and replay after new input data: exact;
- per-call overhead (256-element bf16, median of 1000 enqueues, two rounds): Mojo cuModule 2.88 / 2.74 us, nvcc
  `<<<>>>` 2.93 / 2.83 us, torch.add 2.69 / 2.69 us (p99 3.0-3.3 us; back-to-back 2.7-2.9 us per call). Launch cost
  is the driver's, the Mojo-compiled kernel adds nothing over nvcc's.

## Blockers and caveats for a Mojo port of TensorFold's CUDA kernels on GB10

- No blocker found in the toolchain or launch path; every instruction family the kernels use (HMMA bf16, LDSM,
  LDGSTS, QMMA e4m3, OMMA fp4) is reachable and correct on one tile.
- The kernels still have to be written for Ampere-style `mma.sync` + `cp.async` (no tcgen05/TMEM on GB10); MAX's own
  library kernels for FP8/NVFP4 are sm_100-gated, so nothing there carries over (ours would be hand-written, as on AMD).
- fp4 block-scaled mma has no Mojo wrapper: inline PTX with an asm-internal store (a first attempt, an fp8 `mma.sync`
  as a 4-output `inlined_assembly` returning SIMD[f32,4], crashed the NVPTX instruction selector; `mma()` for fp8 and
  the store-inside-asm form for fp4 both work). Its fragment layout still has to be verified.
- TMA bulk-tensor copies are importable but untested, and `cp.async.bulk` needs a host-built CUtensorMap.
- Mojo has no compile-time arch name for the GB10 other than `sm_121`; objects embed PTX, so a build step with ptxas
  (CUDA 13.x) or the driver's JIT is part of the launch path (cache the cubin by source hash, as `cuda/mojo.py` does).
- Bandwidth is only a 16-byte-load read stream; nothing was run at occupancy-limited or mixed read/write patterns.
- Not checked: MAX CustomOpLibrary/path A on this box, multi-GPU, Mojo's behavior while vLLM `retro` was loaded
  beyond the idle GPU (it was up and idle, 0% busy).

## Files

setup.sh, vecadd.mojo, intr.mojo/.py (a-d), fp8.mojo/.py, fp4.mojo/.py (e), sass.sh (PTX + cubin + SASS grep), bw.mojo/.py,
run_all.sh, results.txt; ../cumodule/ (kernels.mojo, mojo2cubin.py, bench.py, run.sh).
