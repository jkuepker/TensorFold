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
