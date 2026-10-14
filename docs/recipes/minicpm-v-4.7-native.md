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

## Kernels

On a GB10 (sm_121) the Qwen3.6 engine runs on the kernels built into the executable, like Nemotron, and needs no
captured Triton set: its norms, Gated DeltaNet glue, attention prep, tree attention, router and combine are our CUDA
kernels (`qwen_glue.cu`, `qwen_prep.cu`, `qwen_attention.cu`, `qwen_route.cu`), byte-equal to the Triton launches the Python
0.6.6 engine makes. Leave `TENSORFOLD_CUDA_KERNELS` unset (or set it to `native`). Another chip refuses to open with
`QwenNeedsCapturedKernels`, because the built-in kernels were verified on sm_121 only.

A captured set is still accepted: `TENSORFOLD_CUDA_KERNELS=<folder>` (or a bundled `share/tensorfold/cuda/sm121`) names one,
and on a GB10 the built-in kernels still take every launch they cover. The built-in kernels cover Qwen3.6-35B-A3B's
geometry (hidden 2048, 16 key and 32 value heads of 128, 16 query and 2 key-value heads of 256, 256 routed experts
with 8 picks, 8192 conv channels). A checkpoint of another geometry is refused at load with `QwenNeedsCapturedKernels` when
there is no set, and with a set its uncovered launches replay from it. The set is produced as
`zig/tests/cuda/qwen36/README.md` describes.

## Serve

```bash
tensorfold-native serve ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit --name minicpm --parallel 2 --context 49152 --no-thinking
```

Thinking is on by default in the chat template and `max_tokens` counts it; `--no-thinking` or `chat_template_kwargs:
{"enable_thinking": false}` turns it off, and every figure below is with it off. Startup on the GB10, at a 49,152-token
window: the model with its scratch takes 20.02 GiB (20.56 GiB with the 558 MiB draft head), a stream takes 1.00 GiB
(1.09 GiB with drafts), and a kept prompt state takes 62 MiB plus 20 KiB a prompt token (24 KiB with drafts). Load took
7 to 31 s, the longer ones reading the weights from disk.

## Toolkit and driver

Two settings outside the code move single-stream decode on a GB10, and neither changes an output byte.

- nvcc 13.3 builds the fatbins (`-Dnvcc=`). Its SASS for the expert and `qmm_group` kernels is instruction for instruction
the Python 0.6.6 extension's (the NGC image builds it with 13.3); nvcc 13.0 emits a different expert kernel (192
registers and a stack frame where 13.3 has 194 and none) that decodes about 1% slower. Both pass every exactness check.
- The server opens `libcuda.so.1`, and the CUDA 13.3 compat `libcuda` from the NGC PyTorch image (under
`/usr/local/cuda/compat/lib`, on the unchanged 580 kernel driver) closes the gaps between launches that the stock 580
`libcuda` leaves: the GPU is busy 98% of a serial token instead of 92%. Put a folder holding it first on the loader path:

```bash
LD_LIBRARY_PATH=<folder with the compat libcuda.so.1> tensorfold-native serve ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit ...
```

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

These rows were rerun on the built-in kernels with no captured set (`TENSORFOLD_CUDA_KERNELS` unset): the first three
rows pass `serve_check.py` unchanged (the fourth was not rerun). The drafted `--parallel 1` run of the earlier captured-set build accepted 41 of 68, 78 of 140, 22 of 50, 15 of 16, 123 of 297 and 16 of 18 drafted tokens
on the six replies. Qwen3.6's own `tf-cuda-test qwen36-layers` stays byte-equal to the Python engine's dumps (all 40
layers, four prompts) with the vocabulary taken from the config. The tool-call smoke test (a tool call, the same call
streamed, newlines in a plain answer) passes 3 of 3 on the drafted server.

## Measurements

One DGX Spark (GB10) held exclusively (the vLLM server of about 64 GB resident beside it), thinking off,
drafts on unless the row says otherwise, `localeval speed --thinking off --concurrency 2` (3 reps of 128 forced tokens,
median). The native server ran `--parallel 2 --context 49152` on the built-in kernels alone (no captured set), built with
nvcc 13.3, with the CUDA 13.3 compat `libcuda` on the loader path (see Toolkit and driver). Decode tok/s at the prompt
length of the column; the 2-stream figure is the aggregate of two streams at 1k; prefill is the 16k row's uncached
prefill tok/s. Each native row is the mean of two sweeps run back to back, because one sweep's decode spread was 10 to
30%.

| | 0.1k | 1k | 4k | 16k | 2 streams | prefill @16k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native, MiniCPM, drafts | 151.0 | 153.7 | 139.3 | 133.6 | 154.2 | 5,446 |
| Native, MiniCPM, `--no-drafts` | 91.9 | 90.1 | 87.9 | 77.2 | 117.5 | 6,888 |
| Native, stock Qwen3.6, drafts | 157.7 | 198.7 | 160.0 | 147.2 | 169.3 | 5,485 |
| Python 0.6.6, MiniCPM, drafts | 154.8 | 153.2 | 147.6 | 142.1 | 119.6 | 6,599 |
| Python 0.6.6, MiniCPM, serial | 92.1 | 89.7 | 87.5 | 79.6 | 80.9 | 7,033 |
| vLLM, NVFP4 W4A16 (attention, lm_head), Marlin | 99.4 | 102.5 | 101.3 | 93.1 | 115.1 | - |

The sweeps behind each native row, 0.1k / 1k / 4k / 16k / 2 streams: MiniCPM drafts
`20261013-201033-adhoc-ab-new-native-minicpmv47-drafted` (151.1, 153.5, 134.2, 135.0, 152.1) and
`20261013-201100-adhoc-ab-new-native-minicpmv47-drafted-r2` (150.8, 153.9, 144.4, 132.2, 156.3); `--no-drafts`
`20261013-201146-adhoc-ab-new-native-minicpmv47-nodrafts` (92.0, 90.3, 88.1, 77.4, 116.3) and
`20261013-201218-adhoc-ab-new-native-minicpmv47-nodrafts-r2` (91.7, 89.9, 87.8, 77.0, 118.7); Qwen3.6 drafts
`20261013-201320-adhoc-ab-new-native-qwen36-drafted` (157.5, 157.9, 163.1, 146.2, 177.1) and
`20261013-201346-adhoc-ab-new-native-qwen36-drafted-r2` (157.8, 239.6, 156.9, 148.2, 161.5). The Qwen3.6 1k figure is not stable: the first sweep's three reps ran from 147.5 to 250.4
tok/s and the second's from 158.0 to 241.2 (two of them near 240), so the 198.7 in the table is a mean of two medians that
sit far apart; the reps of a drafted row differ by up to 2x. The Python rows are
`20261013-005703-adhoc-tf-minicpmv47-mlx4-qwenmtp` and `20261013-012144-adhoc-tf-minicpmv47-mlx4-serial`, and vLLM's are
from the Python recipe's table (another day's run).

The same session also swept the previous build (this branch before its kernels went native, run with the captured Triton
set rebuilt for programmatic dependent launch, same toolkit and driver) as a control: MiniCPM drafts
`20261013-201433-adhoc-ab-base-d7ca-minicpmv47-drafted` and `-r2` gave a mean of 143.8, 151.2, 146.0, 131.4 and 152.6
(prefill @16k 5,461), and `--no-drafts` `20261013-201544-adhoc-ab-base-d7ca-minicpmv47-nodrafts` and `-r2` gave 90.0, 87.5,
86.0, 77.9 and 114.6 (6,910). Against it the built-in kernels are within -4.6% to +5.0% on drafts (the 4k column, 139.3
against 146.0, is the low one, and its two sweeps were 134.2 and 144.4) and within -0.9% to +3.0% on `--no-drafts`. The
median of a 3-rep sweep moves by 10% or more between runs, so a difference under about 10% is noise, and no column is
beyond it. Against the table this page held before (captured set, nvcc 13.3: drafts 147.4, 149.2, 142.3, 132.8, 148.6;
`--no-drafts` 89.6, 87.3, 86.0, 78.3, 114.5) no column is lower by more than 2.1% (4k drafts) or 1.4% (16k `--no-drafts`).

Native drafted decode is at 94 to 100% of the Python engine's drafted decode in the four single-stream columns (and above
it at two streams), native serial decode is at 97 to 101% of Python's serial decode, and native drafted decode is 1.6 to
1.7 times its own serial decode. The native engine launches each op separately and the Python engine replays CUDA graphs;
the fused projections, the 0.3 confidence rule for ending a draft chain (the same as Python 0.6.6's) and the dependent
launches closed most of that gap, and what is left is about the spread between sweeps. Against vLLM's NVFP4 server, which
does not draft, native drafted decode is higher in every column (for example 133.6 against 93.1 at 16k and 154.2 against
115.1 at two streams), and native serial decode is lower (77.2 against 93.1 at 16k). The vLLM figures come from another
day's run.

Quality, `localeval quality --bench mmlu --limit 25 --max-tokens 2048 --judge none --thinking off`, 1,425 questions:

| | MMLU |
| --- | ---: |
| Native, MiniCPM, drafts (`20261013-132914-adhoc-native-minicpmv47-drafted`) | 83.6 |
| Python 0.6.6, MLX 4-bit | 83.6 |
| vLLM, NVFP4 W4A16 | 83.1 |

## Not done

- TODO: the expert kernels run about 8% longer a launch than the Python engine's identical SASS (gate/up 58.7 against
54.0 us at one row); the allocator, module loading, nvcc and `libcuda` are not the cause.
- TODO: CUDA graphs on the native engine (single-stream decode with drafts is 3 to 7% under the Python engine's).
- TODO: images and `--vision` (upstream #598).
- TODO: native GSM8K, IFEval and HumanEval, 32k and long-context code runs, memory at 262,144 tokens.
- TODO: `--prefill-fp8`; the native prefill is bf16.
- TODO: more than two streams measured on MiniCPM; Qwen3.6 serves up to 16.
