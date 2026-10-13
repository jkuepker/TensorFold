# MiniCPM-V 4.7 35B-A3B

The `minicpmv4_7` family serves MiniCPM-V 4.7 on one NVIDIA GPU, text and images. Its language model is a fine-tune of
Qwen3.6-35B-A3B (`text_config.model_type` `qwen3_5_moe_text`, a 248,144-id vocabulary), so the text side runs on the
[Qwen3.6 MoE CUDA engine](qwen3.6-moe.md) unchanged: Gated DeltaNet and gated attention, routed experts, MTP chains
verified exactly, CUDA graphs per width and context bucket. `--vision` adds MiniCPM's own image tower. The family has
no Mac engine: `tensorfold serve` on a Mac refuses it.

## Checkpoint

There is no published MLX checkpoint yet. Convert openbmb's BF16 release with `tools/minicpmv47_to_mlx.py`, using the
Python that has mlx-lm (on a Mac, or anywhere mlx-lm installs):

```bash
hf download openbmb/MiniCPM-V-4.7 --local-dir ~/models/minicpm-v47-bf16
python tools/minicpmv47_to_mlx.py --src ~/models/minicpm-v47-bf16 --out ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit
python tools/minicpmv47_mlx_check.py inventory --out ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit   # optional checks
```

The converter runs `mlx_lm.convert` on a text-only view of the checkpoint, so the language weights get mlx-lm's own
Qwen3.6 rules: experts split into gate, up and down projections, norms shifted by one, projections at 4 bits in groups
of 64, the router and shared-expert gate at 8 bits. The text shards are byte for byte the layout of
`mlx-community/Qwen3.6-35B-A3B-4bit` apart from the vocabulary. The vision tower (`model.vision_tower.*`, 453
tensors) and merger (`model.merger.*`, 6) stay BF16 in `vision-bf16.safetensors`, `config.json` stays MiniCPM's own
(top-level `model_type: minicpmv4_7`) plus the `quantization` block, and the tokenizer, chat template and processor
files are copied. About 20 GB.

### Drafts: Qwen3.6's MTP layer

MiniCPM-V 4.7 ships no MTP layer. Its language model is close enough to Qwen3.6-35B-A3B (the router gates sit about
twice as close to Qwen3.6's as to Qwen3.5's) that Qwen3.6's MTP layer drafts it well, and the side file holds no
embedding or output head, so the vocabulary difference does not matter: the head reads the target's final normed
state and MiniCPM's own embedding, and its logits go through MiniCPM's head. Drafts are verified against the target's
keyed draw like any MTP chain, so replies equal `"draft": false` whatever the head proposes; a worse head only costs
speed. On code, 71 of 75 drafted tokens were accepted.

The engine takes the first of: `TF_QWEN36_MTP=/path/mtp-4bit.safetensors` (`0` for none), a `mtp-4bit.safetensors`
beside the checkpoint, or the cached `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` file, and prints which it used:

```bash
hf download TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP mtp-4bit.safetensors      # 0.5 GB, once
```

Without one it refuses to start unless `--no-drafts` is given.

## Serve

```bash
tensorfold serve ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit --name minicpm --context 49152 --vision
```

Without `--vision` the tower is not loaded and its tensors are left out of the memory estimate. Text requests are
served exactly as on Qwen3.6 MoE, prompt states kept at message starts. `--no-drafts` or `"draft": false` decodes
serially; drafted replies equal those.

### Images

`--vision` loads the tower and merger in BF16 (about 0.9 GB) next to the text model and reserves 1 GiB for a tower
call's activations (`--vision-offload`: the tower stays in host RAM and visits the GPU per request, 2 GiB reserved).
It runs in plain PyTorch, no transformers needed on the server. Requests send OpenAI `image_url` parts as in
[images](../vision.md).

- **Slicing.** Each image becomes a thumbnail of at most 448 x 448 pixels plus, for a larger image, up to 9 slices of
  about that size (MiniCPM's processor, `max_slice_nums` from the checkpoint's processor config). A slice or thumbnail
  is 14-pixel patches; the tower's window merger after encoder layer 6 and the 2 x 2 merger fold 16 patches into one
  language-model token, so a full slice is 64 tokens and a 1280 x 800 screenshot about 440. The prompt layout is
  MiniCPM's: `<image_id>N</image_id><image>` thumbnail pads `</image>`, then each row of slices as
  `<slice>` pads `</slice>` blocks, rows joined by newlines. The request's images share `--vision-image-tokens`
  (default 4,096); an image over its share is sliced less, down to its thumbnail, and `detail: low` sends the
  thumbnail alone.
- **Positions.** MiniCPM-V 4.7 places image tokens with its canvas 3-D rotary positions: the thumbnail is stretched
  over the slice canvas, each slice sits at its row and column offset, the markers and row newlines pin the canvas
  corners and edges, and text after the image goes on from one past the canvas's larger side. Images separated only by
  markers and newlines are laid out frame by frame, as the reference does. The language model reads them through the
  interleaved `mrope_section` (11, 11, 10) of its attention layers; DeltaNet layers take no positions. Decoded tokens
  continue at their index plus the prompt's rope delta (the largest prompt position + 1 - prompt length), in the MTP
  chains' CUDA graphs too. The positions are vLLM's and transformers' `get_rope_index` exactly (tests compare them).
- **Caching.** Image requests prefill from a fresh state and keep no prompt states; text requests keep prefix
  caching.

On one DGX Spark, greedy with thinking off: the large-digit probe (seven 1280 x 800 renders of "4729" and "8316"
at 40-160 px, 488 prompt tokens each) read 7 of 7, as vLLM does; a drawn red circle, blue square and green triangle
came back as "a red circle, a blue square, and a green triangle". Image replies of 235 and 300 tokens equal their
`"draft": false` runs token for token, and a text request after them still gives its text-only reply. Time to first
token with the tower is 0.3-0.4 s for a 1280 x 800 image. `--vision-offload` reads all seven too, but copies the tower
in for every request (2-7 s on GB10, whose GPU shares host memory anyway), so leave it off there.

## Measurements

One DGX Spark (GB10), NVIDIA's `pytorch:26.07-py3` container, checkpoint converted as above, Qwen3.6's MTP side file,
`--context 49152`, `localeval speed --thinking off --concurrency 2` (decode tok/s at about 0.1k, 1k, 4k and 16k-token
prompts, 128 forced tokens, median of three; two streams aggregate):

| | 0.1k | 1k | 4k | 16k | 2 streams | prefill @16k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorFold, text-only view of the checkpoint, drafted | 154.8 | 153.2 | 147.6 | 142.1 | 119.6 | 6,599 tok/s |
| TensorFold, `--vision`, drafted (two runs) | 162.5 / 162.2 | 162.6 / 168.7 | 150.6 / 144.5 | 123.0 / 139.7 | 113.5 / 123.7 | 6,452 / 6,474 tok/s |
| TensorFold, `--no-drafts` | 92.1 | 89.7 | 87.5 | 79.6 | 80.9 | 7,033 tok/s |
| vLLM 0.31.1rc1, NVFP4A16 (attention and head BF16) | 99.4 | 102.5 | 101.3 | 93.1 | 115.1 | |
| vLLM 0.31.1rc1, NVFP4A16, DSpark k=4 | 117.4 | 118.2 | 115.3 | 62.6 | | |

`--vision` leaves text decoding as it was: the two runs scatter around the text-only one (localeval flagged 14-26%
spread on the first). Long-context code (18 tasks, speed set): 176.6 / 166.9 / 141.5 tok/s at 4k / 16k / 32k, 161.7
on average (vLLM NVFP4A16: 92.0). Quality, text-only, drafted: MMLU 83.6 (1,425 questions), GSM8K 48 (25), IFEval 88
(25), HumanEval 97.5 (40); vLLM NVFP4A16 83.1 / 48 / 92 / 100.

The startup estimate is 26.8 GiB without `--vision` and 28.8 GiB with it (tower and its 1 GiB workspace).

TODO(josef): the vLLM serve commands, and any image-throughput numbers you want here.

## Caveats

- CUDA, one GPU. No Mac engine (the MLX path refuses the family), no `--tp 2`.
- Images only: video is refused, and so is `downsample_mode` 4x and per-request `mm_processor_kwargs`.
- Image input decodes one stream at a time: `--vision` with `--parallel` above 1 is refused at startup. Text-only
  servers take `--parallel` as on Qwen3.6 MoE.
- Several images in one request work (each with its own `<image_id>`, the canvas positions laid out per image), up
  to `--vision-max-images`; they are encoded in runs of at most 16,384 patches.
- The image processor resizes with Pillow's bicubic filter; transformers' default torchvision path can differ by one
  8-bit level per pixel. Token ids, slice grids and positions match it exactly.
- Tool calls use the same `<tool_call>` parser as Qwen3.6.
