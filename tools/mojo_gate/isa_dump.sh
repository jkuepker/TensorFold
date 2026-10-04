#!/bin/sh
# Emit gfx1201 ISA (.amdgcn sidecar per kernel) and grep the instructions of interest.
mkdir -p isa
mojo build --emit asm --target-accelerator gfx1201 intr.mojo -o isa/intr.s
grep -H -o -E "v_wmma[a-z0-9_]*|global_load_tr[a-z0-9_]*|v_dot2_f32_bf16|ds_(load|store)_[a-z0-9_]*|th:TH_LOAD_NT[A-Z_]*" isa/*.amdgcn | sort | uniq -c
