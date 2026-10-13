# Qwen3.6 MoE engine against the Python engine

`capture.py` runs the Python CUDA engine on stock Qwen3.6-35B-A3B MLX 4-bit and dumps, per prompt: every
norm's `h`, `y` and `xs`, every MoE layer's router logits, picks, weights and output, each step's logits and the
state a prefill commits (the layout is in its docstring). Two runs of it hash equal.

1. Dumps and the Triton launches, in the Python engine's container, on a copy of its tree:
   `TRITON_CACHE_DIR=<cache> PYTHONPATH=<copy>/src python3 capture.py --model <checkpoint> --out <dumps> --tools <folder of triton_aot_manifest.py>`
2. The kernel set, as `nemotron/box/README.md` steps 2 and 3 (`triton_aot_manifest.py --launches <dumps>/launches.json`, then `nemotron/aot_pack.py --jit <dumps>/jit.json`).
3. The Zig engine against the dumps, byte for byte:
   `tf-cuda-test qwen36-layers <checkpoint> <kernel set> <dumps> [story,code,facts,long] [--steps N] [--teacher]`

The test runs the whole 40-layer engine (prefill, one-row steps from an empty state, steps after the prefill)
and compares at every norm and every MoE layer, so the first differing op is the first line it prints.
`--teacher` puts the dump's bytes in place of each compared value, so one layer's fault does not flow on.
