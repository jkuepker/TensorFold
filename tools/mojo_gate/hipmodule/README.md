# Phase 1a: launch a Mojo-compiled gfx1201 kernel from torch via hipModuleLaunchKernel

Result: works, at HIP-like overhead (see results.txt). Run on card B (pci_bus_id 7), container
`mojo-p1a` (same pinned image and ~/mojo-gate venv as Phase 0), fleetq job 455.

## Code-object route that works
`mojo build --emit object --target-accelerator gfx1201 kernels.mojo -o k.o` leaves one complete
AMDGPU ELF (EM_AMDGPU 224, ET_DYN, gfx1201, with the AMDGPU metadata note) per instantiated kernel
embedded in the host object (`--emit shared-lib` embeds the same). `mojo2hsaco.py` carves them out
(scan for `\x7fELF` with e_machine 224, size = end of the last section/shdr table), writes
`<kernel>.hsaco` and `manifest.json` (symbol, kernarg size, explicit and hidden args). The blobs
load as-is with `hipModuleLoadData`. Every kernel must be instantiated by a host function
(`enqueue_function[k]`); `kernels.mojo` has an exported `gate_all` for that (never called).

Not usable as-is: assembling the `--emit asm` `.amdgcn` sidecars with ROCm 7.14 clang
(`-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201`) fails: the `.amdgcn_target` id says
`amdgcn-amd-amdhsa-unknown-gfx1201` (clang wants `--gfx1201`) and `.amdhsa_inst_pref_size
((instprefsize(...)<<4)&4080)>>4` does not parse. Not pursued further, route 1 works.

## Kernarg layout (manifest.example.json)
- Explicit args are packed at natural alignment in declaration order, no padding surprises:
  pointer 8 B, Int32 4 B, Int64 8 B (argprobe: ptr@0, i32@8, i64@16, i32@24, i64@32, size 40).
- Hidden args: only kernels that read the launch geometry (global_idx -> scale_add) get them.
  scale_add: explicit 28 B, then hidden_block_count_x/y/z @32, hidden_group_size_x/y/z @44,
  hidden_remainder_x/y/z @50, hidden_global_offset_x/y/z @72, hidden_grid_dims @96; kernarg size
  288. wmma_kernel (thread_idx only) has none: size 24.
- With `kernelParams` (explicit args only) HIP fills the hidden args from the metadata. With
  `HIP_LAUNCH_PARAM_BUFFER` (extra) you build the buffer yourself, hidden args included.

## Files
kernels.mojo, mojo2hsaco.py, bench.py (torch extension via load_inline + checks + timing),
run.sh, results.txt, manifest.example.json.
