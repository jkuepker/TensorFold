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

Served: `tensorfold-native serve MODEL --no-drafts` (with `TENSORFOLD_CUDA_KERNELS` set to the captured kernel set)
answers like the Python 0.6.6 engine run serial. `serve_check.py record URL FIXTURE` records a Python server's prompt
ids (/tokenize), greedy reply ids (`return_token_ids`, temperature 0, 256 tokens, thinking off) and parsed tool calls;
`serve_check.py check URL FIXTURE` reruns the fixture against any server and compares the chat template's ids (a), the
reply ids (b) and the tool calls (c). Stdlib Python only.

## MTP drafts

The checkpoint's MTP layer drafts through the lane core (`families/qwen3_5/cuda/mtp*.zig`): the kept rows' final normed
states are absorbed into the head, which then chains up to three drafts, ending a chain after a draft it gives under 0.3
probability (the Python engine's rule); the target verifies them in one pass, the
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

## Prompt cache

The native CUDA server keeps a stream's state (each DeltaNet layer's recurrence and conv window, each
attention layer's key and value rows) at the cuts the lane core names, 62 MiB and 20 KiB a prompt token (the startup
line says so). `cache_check.py record URL FIXTURE` against a server run with `--prompt-cache-gib 0` records three
conversations turn by turn (a short chat, a three-turn chat, a ~5k-token history, fixed assistant texts between the
turns); `cache_check.py check URL FIXTURE` against a server with the cache on must give the same reply ids token for token,
a hit (usage's `cached_tokens`) on every turn after a conversation's first, and `/metrics`' `prefix_cache_hits_total`
equal to the hits. `serve_check.py check` against a cache-on server is the same check against the Python engine.

With drafts on, a drafting stream's kept state also holds its prompt's final normed rows (4 KiB a token more), so the
draft head reads the whole prompt after a restore; a state kept by a `"draft": false` stream has no rows and a drafting
stream runs its prompt whole instead (the entry stays). `cache_check.py check` passes against a drafted server too.

## Concurrent streams

`--parallel N` (up to 16) serves N streams: every stream's window goes through one shared round (the 27B's
multi_tree_forward with the MoE block on the round's rows), drafted or not, each keeping its own state, head cache and
keyed draws, so a reply equals the same request alone. Prompts still run one at a time through the chunked prefill.

1. The glue kernels' MULTI variants and the row-count alignments of shared rounds come from the Python engine run with
   N streams: `capture.py --lanes 8 --model <checkpoint> --out <dir> --tools <triton tools>` (`capture_multi.py` is its
   body; it sends N requests at once, greedy and seeded sampled, drafted and not). Pack its manifest with the MTP run's
   (`aot_pack.py` takes `--manifest` and `--cache` in pairs) into the kernel set the server loads.
2. `lanes_check.py URL FIXTURE ORACLE [1,4,8]` against a server with drafts on and `--parallel` of at least the largest
   count: each request alone first, then N at once (the 5k-token prompt and the seeded sampled request among them),
   then N at once with every second one sent with `"draft": false`; every reply must equal its solo reply token for
   token (the solo ones also equal the Python engine's serial replies).

## MiniCPM-V 4.7 (text)

`../minicpm/fixture.json` holds `serve_check.py record` output from a Python 0.6.6 MiniCPM-V 4.7 server (serial, drafts
off, thinking off): six chat requests with their prompt ids, greedy reply ids and parsed tool calls. The native server
(`minicpmv4_7`) answers it token for token, drafted and with `--draft-off`, alone and with two clients at once:
`serve_check.py check URL ../minicpm/fixture.json [--draft-off]`. It uses the same captured kernel set as Qwen3.6.

## Programmatic dependent launch

Each plain launch on the stream leaves the GPU about 2 us idle before the next kernel starts; a kernel launched with
the PDL attribute is staged while the one before it still runs, and the gap closes. The native kernels that follow
another grid (`qwen_route.cu`'s three, `gdn.cu`'s tree and replay) begin with `griddepcontrol.wait` and are launched
that way on GB10. The captured Triton kernels have no wait, so `aot_pdl.py --set <kernel set> --cache <Triton cache>
[--cache ...] --ptxas <ptxas> --out <new set>` rebuilds a set: it finds each kernel's PTX by hash in the Triton
caches, builds it with Triton's own ptxas command and requires the captured cubin back byte for byte (use the ptxas of
the container that ran the capture; run the script inside it), then builds it again with `griddepcontrol.wait` as the
first instruction and marks the kernel `pdl` in `aot.json`. The loader launches `pdl` kernels with the attribute. The
rebuilt SASS has the same instructions plus the wait (`ACQBULK`, ahead of every memory access); on some kernels ptxas
also renames registers or reorders independent instructions around it, so the checks above must pass on the new set,
as they do on the captured one. Point `TENSORFOLD_CUDA_KERNELS` at it.

## Built-in row glue

`qwen_glue.cu` (add_rmsnorm, gated_norm, gate_mul, group_sums, paths) and `embed_rows.cu`'s 4-bit rows replace those Triton launches on a GB10, with the cubins' element-to-thread order, sum trees and `sqrt.approx` / `div.full` / `ex2.approx` primitives. `tf-cuda-test qwen-glue <kernel set>` runs each of the set's entries (pointers an entry lists as unaligned sit 4 bytes off a boundary) beside the native kernel on identical buffers, zero rows, inf and NaN rows, subnormal and wide-range rows, and every row count the engine serves, and compares every output byte. `exact` in `families/qwen27/cuda/glue_native.zig` names the kernels that pass on every shape; a shape outside the tested constants (hidden 2048, 32 value heads of 128, 16 query heads of 256) still replays. `qwen36-layers` and `qwen36-mtp` then run with the native kernels in.
