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

Served: `tensorfold-native serve MODEL --no-drafts --parallel 1` (with `TENSORFOLD_CUDA_KERNELS` set to the captured kernel set)
answers like the Python 0.6.6 engine run serial. `serve_check.py record URL FIXTURE` records a Python server's prompt
ids (/tokenize), greedy reply ids (`return_token_ids`, temperature 0, 256 tokens, thinking off) and parsed tool calls;
`serve_check.py check URL FIXTURE` reruns the fixture against any server and compares the chat template's ids (a), the
reply ids (b) and the tool calls (c). Stdlib Python only.

Prompt cache: the native CUDA server keeps a stream's state (each DeltaNet layer's recurrence and conv window, each
attention layer's key and value rows) at the cuts the lane core names, 62 MiB and 20 KiB a prompt token (the startup
line says so). `cache_check.py record URL FIXTURE` against a server run with `--prompt-cache-gib 0` records three
conversations turn by turn (a short chat, a three-turn chat, a ~5k-token history, fixed assistant texts between the
turns); `cache_check.py check URL FIXTURE` against a server with the cache on must give the same reply ids token for token,
a hit (usage's `cached_tokens`) on every turn after a conversation's first, and `/metrics`' `prefix_cache_hits_total`
equal to the hits. `serve_check.py check` against a cache-on server is the same check against the Python engine.
