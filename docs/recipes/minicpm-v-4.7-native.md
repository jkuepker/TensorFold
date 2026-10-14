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

## Receipts

What `CONTRIBUTING.md` asks of a new backend, measured on commit `fedb84fc` (this branch on top of upstream v1.0.10)
with the native server against the 0.6.6 Python engine, same machine, same checkpoints, greedy, one server process at
a time. Every figure below comes from `tools/bench_openai.py`, `tools/bench_concurrent.py` and `tools/bench_cache.py`
run as written in this section. Speed is quoted only where the native and Python replies have the same token-ID hash.
v1.0.10 has no native `qwen3_5_moe` or `minicpmv4_7` path, so there is no before row; the 0.6.6 Python engine is
the applicable standard here, and vLLM was not rerun.

### Setup

| | |
| --- | --- |
| Native build | commit `fedb84fc`, Zig 0.17.0, `zig build native -Dsm=121` with nvcc 13.3.73 (CUDA 13.3), built-in kernels, no captured set (`TENSORFOLD_CUDA_KERNELS` unset), `libcuda` 610.43.02 from NVIDIA's NGC 26.07 PyTorch image first on the loader path (see Toolkit and driver) |
| Native build, nvcc 13.0 | the same commit built with nvcc 13.0.88 and run on the host's own `libcuda` (driver 580.173.02), no setup beyond the README |
| Python engine | TensorFold 0.6.6 in `nvcr.io/nvidia/pytorch:26.07-py3` (torch 2.13.0a0+nv26.07, Triton 3.7.1, CUDA 13.3), its own CUDA 13.3 userland; Qwen3.6 on the container's 0.6.6 tree, MiniCPM-V 4.7 on that tree plus its MiniCPM-V 4.7 recipe (not in a release) |
| Hardware | NVIDIA DGX Spark, GB10 (sm_121, 48 SMs), 128 GB unified memory (121 GiB visible), 20 Arm cores; the memory is shared with the host, and one other model server (vLLM, NVFP4 Qwen3.6, 56.5 GiB resident, idle, GPU 0% busy) stayed loaded throughout, as in the Measurements above |
| OS | Ubuntu 24.04.5, Linux 7.0.0-1019-nvidia, NVIDIA driver 580.173.02 |
| Checkpoints | MiniCPM-V 4.7: `kuepker/MiniCPM-V-4.7-35B-A3B-MLX-4bit-TensorFold` revision `0aee82655c3bdb7135099cd368af7e92235d066f` (the four weight shards, the vision file and the tokenizer have the repository's SHA-256). Qwen3.6-35B-A3B: `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` snapshot `81169a9bc511a27c1b4eedb77a2cd98ced431847` (shards and tokenizer have the repository's current SHA-256). Drafts for both: `mtp-4bit.safetensors`, SHA-256 `c4d7d795b06eaf39c90520940887d81c71abeda3f04f892c2ef0fbc5a8970eb4`, named by `TF_QWEN36_MTP` on both engines |
| Clients | Python 3.12.3, the standard-library tools of this tree |
| Admission | `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=44` for the native server (the Python engine admits 46.9 GiB on this machine); nothing was raised |

The prompts are rebuilt as follows. Read these files of Python 3.12.3's standard library
(`/usr/lib/python3.12/<name>.py` in the Python container), each followed by one newline, in this order: typing,
argparse, subprocess, inspect, dataclasses, enum, functools, collections/__init__, logging/__init__, threading (the 2k
to 64k prompts, 188,719 tokens, text SHA-256 prefix `7608ce4d2fd551b0`); then tarfile, zipfile/__init__, _pydecimal,
pydoc, asyncio/base_events, http/client, email/_header_value_parser, ssl, socket, re/_parser, ast (the 128k and
262,000-token prompts, 430,794 tokens in all, text SHA-256 prefix `98b8e4a1d1674cef`). Encode the concatenation with
the checkpoint's `tokenizer.json` (`tokenizers` 0.23.1, `add_special_tokens=False`) and take the first N ids; the 64k
array is a prefix of the 128k and 262,000 ones, and MiniCPM-V 4.7 and Qwen3.6 give the same ids for this text. The
JSON arrays' SHA-256 (of `json.dumps` of the list):

| Tokens | SHA-256 |
| ---: | --- |
| 2,048 | `e625ad3e95df6ee4686f1cc63941c09b48c764e86b489014fde7530dfd53dadc` |
| 8,192 | `2ad7a1f7a191411ed7c81fed3698e68461f409ce2e233f9711bcfe1de9afde29` |
| 32,768 | `f6c45699bb307b58e26270b712ec64c125b07dc850b5026988c76193b9a1927d` |
| 65,536 | `400de21c9573c33ae8704c700ac902a3dd05c899b9894414c0496677eee7f37e` |
| 131,072 | `17d41a34c5c2660d5720070dfc183e862892ad0be071b8986e480649f682f5fa` |
| 262,000 | `f870d5baf5df513a4446c7f693a775b512eab60079b36f8ebb50626ac68a7428` |

Each report also records its `fixture_sha256`.

### Commands

Decode and concurrency, once per model and engine, drafted and with `--no-drafts`:

```bash
tensorfold-native serve <model> --name bench --parallel 4 --context 8192 --no-thinking [--no-drafts]
python3 tools/bench_openai.py http://127.0.0.1:8097 bench --tokens 256 --reps 3 --temperatures 0 --output decode.json
python3 tools/bench_concurrent.py http://127.0.0.1:8097 bench --tokens 256 --reps 3 --temperatures 0 --levels 1,4 \
  --alone --serial --output concurrent.json
```

The Python engine ran `tensorfold serve <model> --name bench --parallel 4 --context 8192 --no-thinking [--no-drafts]`.
`--serial` (which runs the alone replies again with `"draft": false`) is on the drafted servers only; on the
`--no-drafts` server every request is serial and `--alone` is the check. Each decode run starts a fresh server, sends
one unrelated short request, then runs the two tools; the whole sweep ran twice with the engine order reversed in the
second round, and the tables show both rounds as `first / second`.

Cold prefill, a fresh server for every cell and the prompt cache off:

```bash
tensorfold-native serve <model> --name bench --parallel 1 --context 66560 --no-thinking --prompt-cache-gib 0
python3 tools/bench_cache.py http://127.0.0.1:8097 bench --tokens 32 --repeats 2 --streams 1 \
  --prompt-file ids-65536.json --expected-prompt-tokens 65536 --require-cold --output cache-65536.json
```

The window was 66,560 tokens for 2k to 64k, 132,096 for 128k and 262,144 for the 262,000-token prompt (repeats 1 for
the last two). Every cell was preceded by the same short warm-up request. Native cold prefill is the median time to
first token of the three rows of one process; each row reports `cache_state: cold` with zero cached tokens. The Python
0.6.6 engine resumes a repeated prompt from the state it kept even with `--prompt-cache-gib 0` (its second and third
rows report cached tokens), so its figure is the median of the first request of three fresh processes (one for 128k
and 262,000). Prefill tok/s is the prompt length over that time to first token, which includes sampling the first
token.

### Exactness

Before any speed was quoted, every cell below was compared by the token-ID hash (`token_sha`) between the native and
the Python replies: 44 of 44 cells are equal (8 single-stream, 8 single-stream on the nvcc 13.0 build, 16 concurrent,
12 cold prefill, each over all its reps and both rounds). In `bench_concurrent`, every concurrent reply equals the
same request alone (`alone`, 6 of 6 at one stream and 24 of 24 at four), and on the drafted servers every alone reply
equals its `"draft": false` run (`serial`, 2 of 2 for each cell). The nvcc 13.3 build passes `zig build test` on this
machine (829 of 835 passed, 6 skipped: they need a Metal device or cluster and other-checkpoint inputs), and the 17
model-free tests of `tools/tests` pass. The Parity section's fixture checks (`serve_check.py`, `mtp_check.py`,
`lanes_check.py`, `cache_check.py`) ran on the nvcc 13.0 build of this commit and all passed; they were not rerun on
the nvcc 13.3 build, whose evidence is the cells above.

### Decode

Greedy, 256 tokens, median of 3 reps, tok/s of the decode span. Prompts are `bench_openai`'s: a 14-token completion
(code) and a 31-token chat turn (chat).

| Model | Mode | Prompt | Native, nvcc 13.3 | Python 0.6.6 | Native / Python | Tokens |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MiniCPM-V 4.7 | drafted | code | 189.8 / 189.6 | 189.8 / 189.4 | 1.001 | equal |
| MiniCPM-V 4.7 | drafted | chat | 142.8 / 142.7 | 141.5 / 141.4 | 1.009 | equal |
| MiniCPM-V 4.7 | `--no-drafts` | code | 92.5 / 92.7 | 92.7 / 92.8 | 0.999 | equal |
| MiniCPM-V 4.7 | `--no-drafts` | chat | 92.5 / 92.8 | 92.4 / 92.6 | 1.001 | equal |
| Qwen3.6-35B-A3B | drafted | code | 212.1 / 213.6 | 212.4 / 212.8 | 1.001 | equal |
| Qwen3.6-35B-A3B | drafted | chat | 155.2 / 156.0 | 161.9 / 162.3 | 0.960 | equal |
| Qwen3.6-35B-A3B | `--no-drafts` | code | 92.9 / 92.7 | 92.1 / 92.7 | 1.004 | equal |
| Qwen3.6-35B-A3B | `--no-drafts` | chat | 92.9 / 92.6 | 92.5 / 92.5 | 1.002 | equal |

The native server is within 1% of the Python engine on seven of the eight rows. Qwen3.6 drafted on the chat prompt is
4% under it (155.2 and 156.0 against 161.9 and 162.3), in both rounds; that is the gap the TODO below names. The same
commit built with nvcc 13.0 and run on the host's own `libcuda`, one round:

| Model | Mode | Prompt | Native, nvcc 13.0 | Native, 13.3 | Python 0.6.6 | Tokens vs Python |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MiniCPM-V 4.7 | drafted | code | 186.5 | 189.8 | 189.8 | equal |
| MiniCPM-V 4.7 | drafted | chat | 140.0 | 142.8 | 141.5 | equal |
| MiniCPM-V 4.7 | `--no-drafts` | code | 89.7 | 92.5 | 92.7 | equal |
| MiniCPM-V 4.7 | `--no-drafts` | chat | 89.8 | 92.5 | 92.4 | equal |
| Qwen3.6-35B-A3B | drafted | code | 209.5 | 212.1 | 212.4 | equal |
| Qwen3.6-35B-A3B | drafted | chat | 153.0 | 155.2 | 161.9 | equal |
| Qwen3.6-35B-A3B | `--no-drafts` | code | 90.0 | 92.9 | 92.1 | equal |
| Qwen3.6-35B-A3B | `--no-drafts` | chat | 89.9 | 92.9 | 92.5 | equal |

The nvcc 13.0 build decodes 1.2 to 3.2% under the nvcc 13.3 build with the compat `libcuda`.

Concurrent requests: `bench_concurrent` starts N requests of one prompt at once (greedy, 256 tokens), tok/s is the
aggregate of the streams, and the bracket holds each stream's rate in the two rounds.

| Model | Mode | Prompt | Streams | Native, nvcc 13.3 | Python 0.6.6 | Native / Python | Tokens | Alone / serial checks |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| MiniCPM-V 4.7 | drafted | code | 1 | 189.8 / 189.8 | 189.4 / 189.3 | 1.002 | equal | alone 6/6, serial 2/2 |
| MiniCPM-V 4.7 | drafted | chat | 1 | 141.3 / 141.5 | 141.6 / 141.5 | 0.999 | equal | alone 6/6, serial 2/2 |
| MiniCPM-V 4.7 | drafted | code | 4 | 407.6 / 403.7 [107.4/106.5] | 397.4 / 395.7 [102.6/102.5] | 1.023 | equal | alone 24/24, serial 2/2 |
| MiniCPM-V 4.7 | drafted | chat | 4 | 303.2 / 301.5 [78.4/78.0] | 308.1 / 308.2 [79.2/79.2] | 0.981 | equal | alone 24/24, serial 2/2 |
| MiniCPM-V 4.7 | `--no-drafts` | code | 1 | 92.6 / 92.7 | 91.8 / 92.6 | 1.005 | equal | alone 6/6 |
| MiniCPM-V 4.7 | `--no-drafts` | chat | 1 | 92.5 / 92.5 | 92.7 / 92.4 | 0.999 | equal | alone 6/6 |
| MiniCPM-V 4.7 | `--no-drafts` | code | 4 | 259.5 / 257.6 [66.2/66.2] | 216.7 / 215.5 [55.1/55.0] | 1.196 | equal | alone 24/24 |
| MiniCPM-V 4.7 | `--no-drafts` | chat | 4 | 261.1 / 261.1 [66.8/66.8] | 219.0 / 219.3 [55.8/56.0] | 1.191 | equal | alone 24/24 |
| Qwen3.6-35B-A3B | drafted | code | 1 | 212.4 / 213.3 | 212.3 / 212.7 | 1.002 | equal | alone 6/6, serial 2/2 |
| Qwen3.6-35B-A3B | drafted | chat | 1 | 156.4 / 157.3 | 162.0 / 162.4 | 0.967 | equal | alone 6/6, serial 2/2 |
| Qwen3.6-35B-A3B | drafted | code | 4 | 444.8 / 445.2 [117.7/117.8] | 421.6 / 419.8 [110.0/109.7] | 1.058 | equal | alone 24/24, serial 2/2 |
| Qwen3.6-35B-A3B | drafted | chat | 4 | 352.5 / 355.0 [91.5/91.7] | 351.1 / 349.9 [90.5/90.2] | 1.009 | equal | alone 24/24, serial 2/2 |
| Qwen3.6-35B-A3B | `--no-drafts` | code | 1 | 92.9 / 92.6 | 92.8 / 92.2 | 1.003 | equal | alone 6/6 |
| Qwen3.6-35B-A3B | `--no-drafts` | chat | 1 | 92.8 / 92.6 | 93.3 / 92.6 | 0.997 | equal | alone 6/6 |
| Qwen3.6-35B-A3B | `--no-drafts` | code | 4 | 258.7 / 258.7 [65.7/65.1] | 211.4 / 211.4 [53.9/53.9] | 1.224 | equal | alone 24/24 |
| Qwen3.6-35B-A3B | `--no-drafts` | chat | 4 | 260.8 / 260.2 [66.7/66.5] | 218.0 / 216.4 [55.6/55.5] | 1.199 | equal | alone 24/24 |

With four streams the native server serves serial decode 19 to 22% faster than the Python engine (the Python engine's
rounds over several streams run eagerly) and drafted decode from 2% under to 6% over it, depending on how many drafts
the prompt keeps.

### Cold prefill

Time to first token and prefill tok/s with the prompt cache off. Tokens equal means the 32 greedy reply tokens have
the same hash on both engines.

| Model | Prompt tokens | Native TTFT | Native tok/s | Native, nvcc 13.0 tok/s | Python 0.6.6 TTFT | Python 0.6.6 tok/s | Native / Python | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MiniCPM-V 4.7 | 2,048 | 0.35 s | 5,882 | 5,793 | 0.38 s | 5,352 | 1.099 | equal |
| MiniCPM-V 4.7 | 8,192 | 1.34 s | 6,105 | 6,038 | 1.22 s | 6,729 | 0.907 | equal |
| MiniCPM-V 4.7 | 32,768 | 6.25 s | 5,246 | 5,177 | 5.84 s | 5,611 | 0.935 | equal |
| MiniCPM-V 4.7 | 65,536 | 15.14 s | 4,329 | 4,248 | 13.55 s | 4,837 | 0.895 | equal |
| MiniCPM-V 4.7 | 131,072 | 41.42 s | 3,165 | - | 37.62 s | 3,484 | 0.908 | equal |
| MiniCPM-V 4.7 | 262,000 | 126.62 s | 2,069 | - | 119.64 s | 2,190 | 0.945 | equal |
| Qwen3.6-35B-A3B | 2,048 | 0.35 s | 5,812 | 5,664 | 0.38 s | 5,382 | 1.080 | equal |
| Qwen3.6-35B-A3B | 8,192 | 1.33 s | 6,161 | 6,066 | 1.34 s | 6,126 | 1.006 | equal |
| Qwen3.6-35B-A3B | 32,768 | 6.27 s | 5,226 | 5,104 | 5.53 s | 5,929 | 0.881 | equal |
| Qwen3.6-35B-A3B | 65,536 | 15.17 s | 4,321 | 4,221 | 13.53 s | 4,844 | 0.892 | equal |
| Qwen3.6-35B-A3B | 131,072 | 41.14 s | 3,186 | - | 37.64 s | 3,483 | 0.915 | equal |
| Qwen3.6-35B-A3B | 262,000 | 125.24 s | 2,092 | - | 119.08 s | 2,200 | 0.951 | equal |

The native figure is the median of a first and two repeated requests in one process; the Python figure is the median
of the first request of three fresh processes, since 0.6.6 resumes a repeated prompt even at `--prompt-cache-gib 0`.
Compared first request to first request, native prefill is 4.4 to 4.6% faster at 2,048 tokens (0.367 and 0.364 s
against 0.383 and 0.381 s), not the 8 to 10% the ratio column shows, and 5 to 12% slower from 8,192 up (the two are
level at 8,192 on Qwen3.6, where one of the three Python processes took 1.71 s against 1.23 s and 1.34 s). The nvcc 13.0 build is 1.1 to 2.5% under the nvcc 13.3 one. The native prefill is the bf16 path;
`--prefill-fp8` is not built.

### Memory

Native startup reports what it reserves; the Python engine reports one estimate. The model figure includes the 0.54
GiB draft head and grows with the window (buffers sized by it), the stream figure is one stream's caches at that
window. The kept prompt state is 62 MiB plus 24 KiB a prompt token with drafts (20 KiB without).

| Server | Window | Parallel | Weights | Per stream | Python estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native, drafted | 8,192 | 4 | 20.24 GiB | 0.23 GiB | - |
| Native, serial | 8,192 | 4 | 19.70 GiB | 0.22 GiB | - |
| Native, drafted | 66,560 | 1 | 20.70 GiB | 1.46 GiB | - |
| Native, drafted | 262,144 | 1 | 22.21 GiB | 5.56 GiB | - |
| Python 0.6.6, drafted | 262,144 | 1 | - | - | 42.60 GiB |
| Python 0.6.6, drafted | 8,192 | 4 | - | - | 23.62 GiB |
| Python 0.6.6, serial | 8,192 | 4 | - | - | 22.82 GiB |
| Python 0.6.6, drafted | 66,560 | 1 | - | - | 31.85 GiB |

| Server | Model | Decode run (window 8,192, 4 streams) | 2,048 | 8,192 | 32,768 | 65,536 (window 66,560) | 131,072 (window 132,096) | 262,000 (window 262,144) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native, drafted | MiniCPM-V 4.7 | 20.86 | 20.98 | 21.13 | 21.74 | 22.55 | 24.68 | 28.94 |
| Native, no drafts | MiniCPM-V 4.7 | 20.31 | - | - | - | - | - | - |
| Python 0.6.6, drafted | MiniCPM-V 4.7 | 19.86 | 19.92 | 20.76 | 22.12 | 22.69 | 25.59 | 36.15 |
| Python 0.6.6, no drafts | MiniCPM-V 4.7 | 18.83 | - | - | - | - | - | - |
| Native, drafted | Qwen3.6-35B-A3B | 20.86 | 20.98 | 21.13 | 21.74 | 22.55 | 24.68 | 28.94 |
| Native, no drafts | Qwen3.6-35B-A3B | 20.31 | - | - | - | - | - | - |
| Python 0.6.6, drafted | Qwen3.6-35B-A3B | 19.87 | 19.90 | 20.76 | 22.07 | 22.76 | 25.66 | 36.31 |
| Python 0.6.6, no drafts | Qwen3.6-35B-A3B | 18.83 | - | - | - | - | - | - |

On this 128 GB machine, with the other server resident, one native process peaks at 28.9 GiB with the 262,144-token
window full, against 36.2 GiB for the Python engine. Below that the native process holds about 1 GiB more at decode
and 2k, 0.4 GiB more at 8k, 0.4 GiB less at 32k and 0.1 GiB less at 64k (22.55 against 22.69). The peaks are the
largest per-process figure `nvidia-smi` showed in samples a second apart, so a shorter spike could be missed. The
peak resident set (`VmHWM`) of the native process is 18.0 GiB, most of it load-time staging (0.35 GiB resident while
serving), and of the Python engine 2.2 GiB. A budget here does not describe a smaller machine.

### Not run

- Sampled decoding (the tools run `--temperatures 0` here), the thinking-on template and images (not served).
- Another GPU, driver or OS: sm_121 on this one machine only, with the one Python container above.
- vLLM and the NVFP4 export on the same commit: the vLLM figures above are another day's run, and its server stayed up
  for another user, so no new figure from it is quoted.
- Quality benchmarks: the MMLU row above is the earlier run, not repeated for this commit.
- The raw reports (every request's token counts, hashes and timings, the server logs and the memory samples) are kept
  with the machine's other test logs, not in this repository.

## Not done

- TODO: the expert kernels run about 8% longer a launch than the Python engine's identical SASS (gate/up 58.7 against
54.0 us at one row); the allocator, module loading, nvcc and `libcuda` are not the cause.
- TODO: CUDA graphs on the native engine (single-stream decode with drafts is 3 to 7% under the Python engine's; the
  Receipts show 4% on Qwen3.6's chat prompt and parity on the other rows).
- TODO: the native prefill trails the Python engine's by 5 to 12% from 8,192 tokens up (Receipts).
- TODO: images and `--vision` (upstream #598).
- TODO: native GSM8K, IFEval and HumanEval, 32k and long-context code runs.
- TODO: `--prefill-fp8`; the native prefill is bf16.
- TODO: more than four streams measured; Qwen3.6 serves up to 16.
