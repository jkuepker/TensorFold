# MiniCPM-V 4.7 35B-A3B on the native CUDA server

`tensorfold-native serve` runs MiniCPM-V 4.7's language model on one NVIDIA GPU, text only. The `minicpmv4_7` family is
a thin wrapper of the [Qwen3.6 MoE](qwen3.6-moe.md) engine: the language model is a fine-tune of Qwen3.6-35B-A3B, so
Gated DeltaNet, gated attention, the 256 routed experts and the shared expert, the lane core with its prompt cache, and
the MTP drafts are Qwen3.6's own code. This page lists what MiniCPM changes. It was built and measured on a DGX Spark
(GB10, sm_121) on TensorFold 1.0.9.

Images are not served here. A request with an image part is refused (`this server accepts text parts only; image, audio
and video inputs are unsupported`), and the image tower waits for upstream #598. The 0.6.6 Python engine serves images:
see its MiniCPM-V 4.7 recipe.

## Checkpoint

The MLX 4-bit conversion of openbmb's release (20 GB), the same folder the Python recipe uses:

```bash
hf download kuepker/MiniCPM-V-4.7-35B-A3B-MLX-4bit-TensorFold --local-dir ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit
```

What the engine reads from it:

- `config.json` has the top-level `model_type: minicpmv4_7` and wraps the language model as `text_config` (`model_type`
`qwen3_5_moe_text`); a wrapper around anything else is refused. The geometry is Qwen3.6's, MLX 4-bit in groups of 64
with 8-bit routers and shared-expert gates.
- The vocabulary has 248,144 ids where Qwen3.6's has 248,320. The embedding and head sizes come from the checkpoint's
config, and the head's stores stop at the vocabulary (it is not a multiple of the 128-row tile).
- The `language_model.*` tensors are read. `model.vision_tower.*`, `model.merger.*` and `vision-bf16.safetensors` are left
unread, so they cost no memory.

## Drafts

MiniCPM-V 4.7 ships no MTP layer. Drafts use Qwen3.6's (`mtp-4bit.safetensors`, 0.5 GB), as the Python engine does. The
head reads the target's final normed state with MiniCPM's own embedding and its logits go through MiniCPM's head, over
Qwen3.6's 79,591-id draft vocabulary (every id is below 248,144). The file is looked up in this order:

1. `TF_QWEN36_MTP`: a path, or `0` for none.
2. `mtp-4bit.safetensors` in the checkpoint folder.
3. The Hugging Face cache's copy of `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` (or its older `Vontra/` cache name), in
   `HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`.

```bash
hf download TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP mtp-4bit.safetensors      # 0.5 GB, once
```

With drafts on and no file found the server refuses to start and says so; `--no-drafts` serves without one. A request's
`"draft": false` decodes serially on a drafting server. A drafted reply equals the `"draft": false` one.

## Captured glue kernels

The native Qwen3.6 engine runs its norms, Gated DeltaNet, attention and router glue as Triton cubins captured from the
Python 0.6.6 engine on the same chip, and `TENSORFOLD_CUDA_KERNELS` names the folder (without it the server refuses to
load: `QwenNeedsCapturedKernels`). The set is produced once, on stock Qwen3.6, as `zig/tests/cuda/qwen36/README.md`
describes: `capture.py` in the Python engine's container records every launch, `triton_aot_manifest.py` and
`aot_pack.py` pack them, and `capture.py --mtp` and `--lanes N` add the drafted rounds' verify widths and the concurrent
rounds' variants. MiniCPM uses that set unchanged.

- A set from the layer capture alone serves `--no-drafts --parallel 1`. With drafts or `--parallel 2` the server stops at
load on a launch with no captured variant (`MissingTritonVariant`).
- `aot_pdl.py` (same README) rebuilds a set so its kernels launch with programmatic dependent launch: the idle gap
between kernels closes, about +3% more decode speed on the GB10 at the same bytes. Optional; the unrebuilt set works.
- TODO: own glue kernels, which would remove the capture step.

## Serve

```bash
export TENSORFOLD_CUDA_KERNELS=<captured kernel set>
tensorfold-native serve ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit --name minicpm --parallel 2 --context 49152 --no-thinking
```

Thinking is on by default in the chat template and `max_tokens` counts it; `--no-thinking` or `chat_template_kwargs:
{"enable_thinking": false}` turns it off, and every figure below is with it off. Startup on the GB10, at a 49,152-token
window: the model with its scratch takes 20.02 GiB (20.56 GiB with the 558 MiB draft head), a stream takes 1.00 GiB
(1.09 GiB with drafts), and a kept prompt state takes 62 MiB plus 20 KiB a prompt token (24 KiB with drafts). Load took
7 to 31 s, the longer ones reading the weights from disk.

## Parity with the Python 0.6.6 engine

`zig/tests/cuda/minicpm/fixture.json` was recorded from the Python 0.6.6 MiniCPM-V 4.7 engine, serial, drafts off,
thinking off: six chat requests (system prompt, multi-turn, tool call, tool result, a 256-token story, a 5,101-token
prompt) with their prompt ids, greedy reply ids and parsed tool calls, plus the thinking-on template's prompt ids.
`serve_check.py check URL fixture.json [--draft-off]` reruns it against any server.

| Native server | prompt ids | reply ids | tool calls |
| --- | --- | --- | --- |
| `--no-drafts --parallel 1` | 7 of 7 equal | 6 of 6 equal | 2 of 2 equal |
| drafted, `--parallel 1` | 7 of 7 equal | 6 of 6 equal | 2 of 2 equal |
| drafted, `--parallel 1`, `"draft": false` on every request | 7 of 7 equal | 6 of 6 equal | 2 of 2 equal |
| drafted, `--parallel 2`, two clients running the fixture at once | 7 of 7 equal, both | 6 of 6 equal, both | 2 of 2 equal, both |

The drafted `--parallel 1` run accepted 41 of 68, 78 of 140, 22 of 50, 15 of 16, 123 of 297 and 16 of 18 drafted tokens
on the six replies. Qwen3.6's own `tf-cuda-test qwen36-layers` stays byte-equal to the Python engine's dumps (all 40
layers, four prompts) with the vocabulary taken from the config. The tool-call smoke test (a tool call, the same call
streamed, newlines in a plain answer) passes 3 of 3 on the drafted server.

## Measurements

One DGX Spark (GB10) with a vLLM server of about 64 GB resident beside it, thinking off, drafts on unless the row says
otherwise, `localeval speed --thinking off --concurrency 2` (3 reps of 128 forced tokens, median). The native server ran
`--parallel 2 --context 49152`. Decode tok/s at the prompt length of the column; the 2-stream figure is the aggregate of
two streams at 1k; prefill is the 16k row's uncached prefill tok/s.

| | 0.1k | 1k | 4k | 16k | 2 streams | prefill @16k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native, MiniCPM, drafts | 130.5 | 123.7 | 112.4 | 109.6 | 133.2 | 5,479 |
| Native, MiniCPM, `--no-drafts` | 80.7 | 77.6 | 76.4 | 70.0 | 106.1 | 6,774 |
| Native, stock Qwen3.6, drafts | 137.5 | 118.9 | 141.6 | 127.0 | 131.7 | 5,386 |
| Python 0.6.6, MiniCPM, drafts | 154.8 | 153.2 | 147.6 | 142.1 | 119.6 | 6,599 |
| Python 0.6.6, MiniCPM, serial | 92.1 | 89.7 | 87.5 | 79.6 | 80.9 | 7,033 |
| vLLM, NVFP4 W4A16 (attention, lm_head), Marlin | 99.4 | 102.5 | 101.3 | 93.1 | 115.1 | - |

The native rows are localeval runs `20261013-135101-adhoc-excl-native-minicpmv47-drafted`,
`20261013-135152-adhoc-excl-native-minicpmv47-nodrafts` and `20261013-134951-adhoc-excl-native-qwen36-drafted`, taken
with the Spark held exclusively; the Python rows are `20261013-005703-adhoc-tf-minicpmv47-mlx4-qwenmtp` and
`20261013-012144-adhoc-tf-minicpmv47-mlx4-serial`, and vLLM's are from the Python recipe's table. Decode spread between
reps is 14 to 22% on the drafted rows and one Qwen3.6 rep at 1k ran at 363 tok/s (that row's median is 118.9), so
differences under 10% are noise. A first, non-exclusive pass of the same three sweeps
(`20261013-132839-adhoc-native-minicpmv47-drafted`, `20261013-132700-adhoc-native-qwen36-drafted`) gave MiniCPM drafted
118.0, 127.1, 117.9, 107.9 and 138.5.

Native drafted decode is at 76 to 84% of the Python engine's drafted decode in the four single-stream columns (and above
it at two streams), native serial decode is at 87 to 88% of Python's serial decode, and native drafted decode is 1.5 to
1.6 times its own serial decode. The native engine launches each op separately; the Python engine replays CUDA graphs.
Against vLLM's NVFP4 server, which does not draft, native drafted decode is higher in every column (for example 109.6
against 93.1 at 16k and 133.2 against 115.1 at two streams), and native serial decode is lower (70.0 against 93.1 at
16k). The vLLM figures come from another day's run.

Quality, `localeval quality --bench mmlu --limit 25 --max-tokens 2048 --judge none --thinking off`, 1,425 questions:

| | MMLU |
| --- | ---: |
| Native, MiniCPM, drafts (`20261013-132914-adhoc-native-minicpmv47-drafted`) | 83.6 |
| Python 0.6.6, MLX 4-bit | 83.6 |
| vLLM, NVFP4 W4A16 | 83.1 |

## Not done

- TODO: CUDA graphs on the native engine (single-stream decode with drafts is 16 to 24% under the Python engine's).
- TODO: images and `--vision` (upstream #598).
- TODO: native GSM8K, IFEval and HumanEval, 32k and long-context code runs, memory at 262,144 tokens.
- TODO: `--prefill-fp8`; the native prefill is bf16.
- TODO: more than two streams measured on MiniCPM; Qwen3.6 serves up to 16.
