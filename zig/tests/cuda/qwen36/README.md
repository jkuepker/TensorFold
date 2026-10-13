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

## MTP drafts

The checkpoint's MTP layer drafts through the lane core (`families/qwen3_5/cuda/mtp*.zig`): the kept rows' final normed
states are absorbed into the head, which then chains up to three drafts; the target verifies them in one pass, the
DeltaNet state replays to the accepted path, and every row is drawn by the target's keyed rule, so a drafted reply equals
the `"draft": false` one, sampled requests included. `"draft": false` on a request turns its drafts off.

1. The Python engine's head calls and drafted rounds, in the container as above: `capture.py --mtp --model <checkpoint> --out <dumps>`
   (`capture_mtp.py` is its body; `mtp_weights.txt` holds the head's tensor digests, `<prompt>/greedy/cNNN` each call's
   states, norms, normed rows and draft logits, `tokens.json` the drafted and serial replies, greedy and one seeded sampled
   request each). It also sweeps every verify width, so the Triton set packed from its launches covers drafted rounds.
2. `tf-qwen35-upload MODEL --mtp --digests` prints the head's tensors as `mtp_weights.txt` holds them.
3. `tf-cuda-test qwen36-mtp MODEL <kernel set> <dumps> [PROMPTS]` replays the head calls in order over its own cache:
   every norm, normed row and draft logit byte for byte.
4. `mtp_check.py URL FIXTURE DUMPS` against a native server with drafts on: the fixture's chat requests and the dumps'
   four prompts (greedy, and the seeded sampled request) drafted, with `"draft": false`, and the Python engine's serial
   reply must all match, with acceptance beside the Python engine's.
