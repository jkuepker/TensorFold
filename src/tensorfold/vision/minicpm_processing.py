"""CPU-only MiniCPM-V 4.7 image preprocessing: slicing, placeholder layout and canvas 3-D rotary positions.

Ports, without importing either: transformers 5.19's ``MiniCPMV4_6ImageProcessorPil`` (the slicing and NaViT patch
packing MiniCPM-V 4.7's ``preprocessor_config.json`` names) and ``MiniCPMV4_7Processor.replace_image_token`` (the
placeholder text), and vLLM's ``minicpmv4_7.py`` canvas M-RoPE (``_compute_canvas_packed`` and its helpers, which
mirror transformers' ``get_rope_index``). Both Apache-2.0. Images only, ``downsample_mode`` 16x.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorfold.server.errors import CONTEXT_LIMIT

TOKEN_DIVISOR = 16          # 16x: the tower's window merger and the merger each fold 2 x 2 patches into one


@dataclass(frozen=True)
class Tokens:
    """The checkpoint's structural ids (config.json) and the tokenizer's marker strings."""

    image: int
    image_start: int
    image_end: int
    slice_start: int
    slice_end: int
    newline: int
    image_text: str = "<|image_pad|>"
    image_start_text: str = "<image>"
    image_end_text: str = "</image>"
    slice_start_text: str = "<slice>"
    slice_end_text: str = "</slice>"
    image_id_start_text: str = "<image_id>"
    image_id_end_text: str = "</image_id>"

    @property
    def canvas(self) -> dict[str, int]:
        """The special ids vLLM's canvas M-RoPE reads."""

        return {"im_start_id": self.image_start, "im_end_id": self.image_end, "slice_start_id": self.slice_start,
                "slice_end_id": self.slice_end, "newline_id": self.newline}


@dataclass(frozen=True)
class PreparedMiniCPMPrompt:
    token_ids: tuple[int, ...]
    pixel_values: tuple[np.ndarray, ...]     # one [C, patch, h * w * patch] array a unit (a thumbnail or a slice)
    target_sizes: np.ndarray                 # [units, 2]: each unit's patch grid (h, w)
    units_per_image: tuple[int, ...]
    position_ids: np.ndarray                 # [3, prompt tokens]: temporal, height, width
    rope_delta: int                          # decoded rows sit at their index plus this
    image_spans: tuple[tuple[int, int], ...]   # each unit's placeholder run [start, end)
    image_hashes: tuple[str, ...]

    @property
    def visual_tokens(self) -> int:
        return sum(end - start for start, end in self.image_spans)


# -- slicing (MiniCPMV4_6ImageProcessorPil) ---------------------------------------------------------------------------


def ensure_divide(length: float, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def find_best_resize(size: tuple[float, float], scale_resolution: int, patch_size: int,
                     allow_upscale: bool = False) -> tuple[int, int]:
    height, width = size
    if (height * width > scale_resolution * scale_resolution) or allow_upscale:
        aspect_ratio = width / height
        height = int(scale_resolution / math.sqrt(aspect_ratio))
        width = int(height * aspect_ratio)
    return ensure_divide(height, patch_size * 4), ensure_divide(width, patch_size * 4)


def get_refine_size(size: tuple[int, int], grid: Sequence[int], scale_resolution: int, patch_size: int,
                    allow_upscale: bool = False) -> tuple[int, int]:
    height, width = size
    grid_y, grid_x = grid
    refine_width, refine_height = ensure_divide(width, grid_x), ensure_divide(height, grid_y)
    best_height, best_width = find_best_resize((refine_height / grid_y, refine_width / grid_x), scale_resolution,
                                               patch_size, allow_upscale=allow_upscale)
    return best_height * grid_y, best_width * grid_x


def get_sliced_grid(size: tuple[int, int], max_slice_nums: int, scale_resolution: int) -> list[int] | None:
    original_height, original_width = size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)
    if multiple <= 1:
        return None
    best_grid, min_error = [1, 1], float("inf")
    for num_slices in (multiple - 1, multiple, multiple + 1):
        if num_slices == 1 or num_slices > max_slice_nums:
            continue
        for num_rows in range(1, num_slices + 1):
            if num_slices % num_rows == 0:
                num_cols = num_slices // num_rows
                error = abs(log_ratio - math.log(num_cols / num_rows))
                if error < min_error:
                    best_grid, min_error = [num_rows, num_cols], error
                elif error == min_error and num_rows > best_grid[0]:
                    best_grid = [num_rows, num_cols]
    return best_grid


def reshape_by_patch(image: np.ndarray, patch_size: int) -> np.ndarray:
    """``[C, H, W]`` -> NaViT's ``[C, patch, H * W / patch]`` (patches in raster order along the last axis)."""

    channels, height, width = image.shape
    patches = image.reshape(channels, height // patch_size, patch_size, width // patch_size, patch_size)
    return patches.transpose(0, 2, 1, 3, 4).reshape(channels, patch_size, -1)


class MiniCPMImageProcessor:
    """The checkpoint's image settings: a thumbnail of at most ``scale_resolution``² pixels and, for a larger image,
    up to ``max_slice_nums`` slices of about that size each."""

    def __init__(self, *, patch_size: int = 14, scale_resolution: int = 448, max_slice_nums: int = 9,
                 slice_mode: bool = True, image_mean: Sequence[float] = (0.5, 0.5, 0.5),
                 image_std: Sequence[float] = (0.5, 0.5, 0.5), rescale_factor: float = 1 / 255, resample: int = 3,
                 downsample_mode: str = "16x", use_image_id: bool = True, **_: Any) -> None:
        if downsample_mode != "16x":
            raise ValueError(f"MiniCPM-V image input reads downsample_mode 16x, not {downsample_mode!r}")
        self.patch_size, self.scale_resolution = int(patch_size), int(scale_resolution)
        self.max_slice_nums, self.slice_mode, self.use_image_id = int(max_slice_nums), bool(slice_mode), use_image_id
        self.mean = np.asarray(image_mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(image_std, dtype=np.float32).reshape(-1, 1, 1)
        self.rescale_factor, self.resample = float(rescale_factor), int(resample)

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "MiniCPMImageProcessor":
        path = Path(model_dir)
        options: dict[str, Any] = {}
        processor = path / "processor_config.json"
        if processor.is_file():
            options.update(json.loads(processor.read_text()).get("image_processor") or {})
        elif (path / "preprocessor_config.json").is_file():
            options.update(json.loads((path / "preprocessor_config.json").read_text()))
        return cls(**options)

    def _normalize(self, pixels: np.ndarray) -> np.ndarray:
        return ((pixels.astype(np.float32) * np.float32(self.rescale_factor) - self.mean) / self.std).astype(np.float32)

    def _resize(self, image, height: int, width: int) -> np.ndarray:
        from PIL import Image

        out = image.resize((int(width), int(height)), resample=Image.Resampling(self.resample))
        return np.asarray(out, dtype=np.uint8).transpose(2, 0, 1)

    def units(self, image, *, max_slice_nums: int | None = None, slice_mode: bool | None = None
              ) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
        """One RGB PIL image -> its units' packed patches, their patch grids, and the slice grid (rows, cols; (0, 0)
        without slices), as the transformers processor computes them."""

        image = image.convert("RGB")
        width, height = image.size
        slices = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        grid = (get_sliced_grid((height, width), slices, self.scale_resolution)
                if (self.slice_mode if slice_mode is None else slice_mode) else None)
        p = self.patch_size
        h, w = find_best_resize((height, width), self.scale_resolution, p, allow_upscale=grid is None)
        units = [reshape_by_patch(self._normalize(self._resize(image, h, w)), p)]
        sizes = [(h // p, w // p)]
        if grid is not None:
            rh, rw = get_refine_size((height, width), grid, self.scale_resolution, p, allow_upscale=True)
            refined = self._resize(image, rh, rw)
            ph, pw = rh // grid[0], rw // grid[1]
            for i in range(0, rh, ph):
                for j in range(0, rw, pw):
                    units.append(reshape_by_patch(self._normalize(refined[:, i:i + ph, j:j + pw]), p))
                    sizes.append((ph // p, pw // p))
        return units, sizes, (tuple(grid) if grid is not None else (0, 0))


def placeholder(tokens: Tokens, index: int | None, sizes: Sequence[tuple[int, int]], grid: tuple[int, int]) -> str:
    """One image's text: ``<image_id>i</image_id>`` (``index``), the thumbnail's pads in ``<image>``, then the slices,
    a row of ``<slice>`` blocks per line (``MiniCPMV4_7Processor.replace_image_token``)."""

    counts = [h * w // TOKEN_DIVISOR for h, w in sizes]
    text = tokens.image_start_text + tokens.image_text * counts[0] + tokens.image_end_text
    rows, cols = grid
    if rows > 0 and cols > 0:
        block = tokens.slice_start_text + tokens.image_text * (counts[1] if len(counts) > 1 else 0) + tokens.slice_end_text
        text += "\n".join(block * cols for _ in range(rows))
    if index is not None:
        text = f"{tokens.image_id_start_text}{index}{tokens.image_id_end_text}" + text
    return text


# -- canvas 3-D rotary positions (vLLM minicpmv4_7.py) ----------------------------------------------------------------


def compute_llm_grid(n_tokens: int, vision_grid_height: int, vision_grid_width: int) -> tuple[int, int]:
    if n_tokens <= 0 or vision_grid_height <= 0 or vision_grid_width <= 0:
        return 0, 0
    vit_total = vision_grid_height * vision_grid_width
    if vit_total == n_tokens:
        return vision_grid_height, vision_grid_width
    if vit_total < n_tokens:
        return 0, 0
    factor = int(round(math.sqrt(vit_total / n_tokens)))
    if factor > 0 and vision_grid_height % factor == 0 and vision_grid_width % factor == 0:
        height, width = vision_grid_height // factor, vision_grid_width // factor
        if height * width == n_tokens:
            return height, width
    ratio = vision_grid_height / vision_grid_width
    width = max(1, int(round(math.sqrt(n_tokens / ratio))))
    height = n_tokens // width
    if height * width == n_tokens and height > 0:
        return height, width
    return 0, 0


def image_bounds(ids: Sequence[int], special: dict[str, int]) -> list[tuple[int, int]]:
    """[start, end) of every unit's pads: after each ``<image>``/``<slice>``, up to each ``</image>``/``</slice>``."""

    starts = [i + 1 for i, t in enumerate(ids) if t in (special["im_start_id"], special["slice_start_id"])]
    ends = [i for i, t in enumerate(ids) if t in (special["im_end_id"], special["slice_end_id"])]
    return list(zip(starts, ends))


def _canvas_layout(thumbnail, slices, sizes):
    t_bs, t_be, t_gi = thumbnail
    thumb_h, thumb_w = compute_llm_grid(t_be - t_bs, *sizes[t_gi])
    if not slices:
        return thumb_h, thumb_w, 0, 0, 0, max(thumb_h, 1), max(thumb_w, 1)
    cols = len(slices)
    for k in range(len(slices) - 1):
        if slices[k + 1][0] - slices[k][1] > 2:
            cols = k + 1
            break
    rows = len(slices) // cols if cols > 0 else 1
    if rows * cols != len(slices):
        rows, cols = 1, len(slices)
    s0_bs, s0_be, s0_gi = slices[0]
    slice_h, slice_w = compute_llm_grid(s0_be - s0_bs, *sizes[s0_gi])
    return thumb_h, thumb_w, slice_h, slice_w, cols, rows * slice_h, cols * slice_w


def _assign_canvas_span(pos, *, thumbnail, span_end, slices, base, sizes) -> int:
    """Place one canvas at ``base`` in ``pos`` [3, N]; the distance to the next canvas base."""

    import torch

    t_bs, t_be, _ = thumbnail
    span_start = t_bs - 1                                   # the <image> token
    n_thumb = t_be - t_bs
    thumb_h, thumb_w, slice_h, slice_w, cols, canvas_h, canvas_w = _canvas_layout(thumbnail, slices, sizes)
    pos[:, span_start:span_end] = base
    halo_lo = max(base - 1, 0)
    pos[1, span_start] = halo_lo
    pos[2, span_start] = halo_lo
    if t_be < span_end:                                     # </image>: the canvas's far corner
        pos[1, t_be] = base + canvas_h
        pos[2, t_be] = base + canvas_w
    if thumb_h > 0 and thumb_w > 0 and thumb_h * thumb_w == n_thumb:
        if slices and canvas_h > 0 and canvas_w > 0:
            h_c = torch.linspace(0, canvas_h - 1, thumb_h).round().long()
            w_c = torch.linspace(0, canvas_w - 1, thumb_w).round().long()
        else:
            h_c, w_c = torch.arange(thumb_h), torch.arange(thumb_w)
        pos[0, t_bs:t_be] = base
        pos[1, t_bs:t_be] = h_c.view(-1, 1).expand(-1, thumb_w).reshape(-1) + base
        pos[2, t_bs:t_be] = w_c.view(1, -1).expand(thumb_h, -1).reshape(-1) + base
    else:
        pos[:, t_bs:t_be] = torch.arange(n_thumb, dtype=torch.long) + base
    for k, (s_bs, s_be, s_gi) in enumerate(slices):
        n_s = s_be - s_bs
        sh_k, sw_k = compute_llm_grid(n_s, *sizes[s_gi])
        h_off, w_off = (k // cols) * slice_h, (k % cols) * slice_w
        if s_bs - 1 >= span_start:                          # <slice>: its top-left corner
            pos[1, s_bs - 1] = base + h_off
            pos[2, s_bs - 1] = base + w_off
        end_h, end_w = (h_off + sh_k - 1, w_off + sw_k - 1) if sh_k > 0 and sw_k > 0 else (h_off, w_off)
        if s_be < span_end:                                 # </slice>: its bottom-right corner
            pos[1, s_be] = base + end_h
            pos[2, s_be] = base + end_w
        if sh_k > 0 and sw_k > 0 and sh_k * sw_k == n_s:
            pos[0, s_bs:s_be] = base
            pos[1, s_bs:s_be] = torch.arange(sh_k).view(-1, 1).expand(-1, sw_k).reshape(-1) + h_off + base
            pos[2, s_bs:s_be] = torch.arange(sw_k).view(1, -1).expand(sh_k, -1).reshape(-1) + w_off + base
        else:
            pos[:, s_bs:s_be] = torch.arange(n_s, dtype=torch.long) + base
    for k in range(len(slices) - 1):                        # "\n" between slice rows: just past the row's right edge
        gap_s, gap_e = slices[k][1], slices[k + 1][0]
        if gap_e - gap_s <= 2:
            continue
        boundary_h = ((k // cols) + 1) * slice_h - 1
        for nl_pos in range(gap_s + 1, gap_e - 1):
            pos[1, nl_pos] = base + boundary_h
            pos[2, nl_pos] = base + cols * slice_w
    return max(canvas_h, canvas_w) + 1


def canvas_positions(ids: Sequence[int], sizes: Sequence[tuple[int, int]], special: dict[str, int]):
    """[3, N] int64 positions of one prompt and its rope delta (vLLM's ``get_mrope_input_positions``): text runs on,
    each image (thumbnail and slices) a canvas at the text position it starts on, a run of images with only markers
    and newlines between them laid out frame by frame. ``sizes``: every unit's patch grid, in prompt order."""

    import torch

    n = len(ids)
    bounds = image_bounds(ids, special)
    if len(bounds) != len(sizes):
        raise ValueError(f"the prompt's {len(bounds)} image units do not match the {len(sizes)} encoded ones")
    pos = torch.arange(n, dtype=torch.long).unsqueeze(0).expand(3, -1).clone()
    if not bounds:
        return pos, 0
    gap_ok = {special[k] for k in ("im_start_id", "im_end_id", "slice_start_id", "slice_end_id", "newline_id")}

    def group_end(g, s_end):
        end = g["slices"][-1][1] if g["slices"] else g["thumbnail"][1]
        while end < s_end and ids[end] in gap_ok:
            end += 1
        return min(end, s_end)

    def tight(g_prev, g_next, s_end):
        return all(ids[p] in gap_ok for p in range(group_end(g_prev, s_end), g_next["thumbnail"][0] - 1))

    raw, current = [], None
    for gi, (bs, be) in enumerate(bounds):
        if bs - 1 >= 0 and ids[bs - 1] == special["slice_start_id"] and current is not None:
            current["slices"].append((bs, be, gi))
        else:
            if current is not None:
                raw.append(current)
            current = {"thumbnail": (bs, be, gi), "slices": []}
    raw.append(current)
    groups, i = [], 0
    while i < len(raw):
        frames, j = [raw[i]], i + 1
        while j < len(raw) and tight(raw[j - 1], raw[j], n):
            frames.append(raw[j])
            j += 1
        groups.append(raw[i] if len(frames) == 1 else {"video_frames": frames})
        i = j
    p, cursor = 0, 0

    def text(a, b, start):
        pos[:, a:b] = torch.arange(b - a, dtype=torch.long) + start

    for group in groups:
        if "video_frames" in group:
            frames = group["video_frames"]
            start, end = frames[0]["thumbnail"][0] - 1, group_end(frames[-1], n)
            if start > cursor:
                text(cursor, start, p)
                p += start - cursor
            frame_cursor = start
            for vf in frames:
                im_start = vf["thumbnail"][0] - 1
                if im_start > frame_cursor:
                    text(frame_cursor, im_start, p)
                    p += im_start - frame_cursor + 1        # a buffer: the next halo misses the gap's last token
                frame_end = group_end(vf, end)
                p += _assign_canvas_span(pos, thumbnail=vf["thumbnail"], span_end=frame_end, slices=vf["slices"],
                                         base=p, sizes=sizes)
                frame_cursor = frame_end
            if frame_cursor < end:
                text(frame_cursor, end, p)
                p += end - frame_cursor
            cursor = end
        else:
            start, end = group["thumbnail"][0] - 1, group_end(group, n)
            if start > cursor:
                text(cursor, start, p)
                p += start - cursor
            p += _assign_canvas_span(pos, thumbnail=group["thumbnail"], span_end=end, slices=group["slices"], base=p,
                                     sizes=sizes)
            cursor = end
    if n > cursor:
        text(cursor, n, p)
    return pos, int(pos.max().item()) + 1 - n


# -- the frontend's CPU half ------------------------------------------------------------------------------------------


class MiniCPMPreprocessor:
    """Images and a rendered chat prompt -> token ids, packed patches and canvas positions, with no model calls."""

    def __init__(self, config: dict, processor: MiniCPMImageProcessor, tokenizer: Any, tokens: Tokens):
        self.config, self.processor, self.tokenizer, self.tokens = config, processor, tokenizer, tokens

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "MiniCPMPreprocessor":
        from tokenizers import Tokenizer

        path = Path(model_dir).expanduser()
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "minicpmv4_7" or not config.get("vision_config"):
            raise ValueError("MiniCPM-V image preprocessing reads a MiniCPM-V 4.7 checkpoint")
        if str(config.get("downsample_mode", "16x")) != "16x":
            raise ValueError("MiniCPM-V image input reads downsample_mode 16x")
        if "canvas" not in str(config.get("mrope_mode", "canvas")).lower():
            raise ValueError("MiniCPM-V image input reads checkpoints with canvas rotary positions (mrope_mode)")
        keys = ("image_token_id", "image_start_id", "image_end_id", "slice_start_id", "slice_end_id", "newline_id")
        if not all(isinstance(config.get(k), int) for k in keys):
            raise ValueError("this MiniCPM-V checkpoint's config.json is missing its image-token ids")
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        names = {}
        settings = path / "tokenizer_config.json"
        if settings.is_file():
            raw = json.loads(settings.read_text())
            names = {k: raw[k] for k in ("image_token", "image_start_token", "image_end_token", "slice_start_token",
                                         "slice_end_token", "image_id_start_token", "image_id_end_token")
                     if isinstance(raw.get(k), str)}
        tokens = Tokens(config["image_token_id"], config["image_start_id"], config["image_end_id"],
                        config["slice_start_id"], config["slice_end_id"], config["newline_id"],
                        names.get("image_token", "<|image_pad|>"), names.get("image_start_token", "<image>"),
                        names.get("image_end_token", "</image>"), names.get("slice_start_token", "<slice>"),
                        names.get("slice_end_token", "</slice>"), names.get("image_id_start_token", "<image_id>"),
                        names.get("image_id_end_token", "</image_id>"))
        for text, wanted in ((tokens.image_text, tokens.image), (tokens.image_start_text, tokens.image_start),
                             (tokens.image_end_text, tokens.image_end), (tokens.slice_start_text, tokens.slice_start),
                             (tokens.slice_end_text, tokens.slice_end)):
            if tokenizer.token_to_id(text) != wanted:
                raise ValueError(f"the tokenizer's {text} is not config.json's id {wanted}")
        return cls(config, MiniCPMImageProcessor.from_directory(path), tokenizer, tokens)

    def encode_text(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False).ids]

    def prepare(self, rendered_prompt: str, images: Sequence[Any], *, videos: Sequence[Any] = (),
                max_visual_tokens: int = 4096, max_prompt_tokens: int | None = None) -> PreparedMiniCPMPrompt:
        """Expand each image marker (the template's ``<|image_pad|>``) into its thumbnail and slices, tokenize, and
        compute the canvas positions. An image is sliced less (down to its thumbnail) to fit the visual-token budget,
        which the request's images share; ``detail: low`` keeps the thumbnail alone."""

        if videos:
            raise ValueError("MiniCPM-V on TensorFold takes images, not video")
        marker = self.tokens.image_text
        if not images or max_visual_tokens < 1:
            raise ValueError("Image preprocessing needs images and a positive visual-token budget")
        if rendered_prompt.count(marker) != len(images):
            raise ValueError("The rendered prompt must contain exactly one image marker for each image")
        share = max_visual_tokens // len(images)
        expanded, units, sizes, per_image = [], [], [], []
        for index, image in enumerate(images):
            low = getattr(image, "detail", "auto") == "low"
            slices = 1 if low else self.processor.max_slice_nums
            while True:
                packed, grids, grid = self.processor.units(image.to_pil(), max_slice_nums=slices,
                                                           slice_mode=False if low else None)
                count = sum(h * w // TOKEN_DIVISOR for h, w in grids)
                if count <= share or slices <= 1:
                    break
                slices -= 1                           # fewer, larger-scale slices: fewer tokens
            if count > share:
                raise ValueError(f"an image takes {count} visual tokens, more than its share ({share}) of the "
                                 f"{max_visual_tokens}-token budget; send fewer images or raise --vision-image-tokens")
            if any(h % 4 or w % 4 for h, w in grids):
                raise ValueError("the image processor returned a patch grid the tower cannot merge")
            expanded.append(placeholder(self.tokens, index if self.processor.use_image_id else None, grids, grid))
            units.extend(packed)
            sizes.extend(grids)
            per_image.append(len(grids))
        pieces = rendered_prompt.split(marker)
        text = pieces[0] + "".join(e + piece for e, piece in zip(expanded, pieces[1:]))
        ids = tuple(self.encode_text(text))
        if max_prompt_tokens is not None and len(ids) > max_prompt_tokens:   # OpenAI's context_length_exceeded
            raise ValueError(f"{CONTEXT_LIMIT} {max_prompt_tokens} tokens: the expanded image prompt has {len(ids)} "
                             "tokens, which exceeds the context window; reduce image resolution or prompt length")
        return self._finish(ids, tuple(units), np.asarray(sizes, dtype=np.int64).reshape(-1, 2), tuple(per_image),
                            tuple(getattr(image, "content_hash", "") for image in images))

    def _finish(self, ids, units, sizes, per_image, hashes) -> PreparedMiniCPMPrompt:
        spans = tuple(image_bounds(ids, self.tokens.canvas))
        counts = [h * w // TOKEN_DIVISOR for h, w in sizes.tolist()]
        if [end - start for start, end in spans] != counts:
            raise ValueError("the tokenized prompt's image placeholders do not match the processed images")
        if any(ids[i] != self.tokens.image for start, end in spans for i in range(start, end)) or \
                sum(counts) != sum(1 for t in ids if t == self.tokens.image):
            raise ValueError("image placeholder tokens appear outside the expanded images")
        positions, delta = canvas_positions(ids, [tuple(s) for s in sizes.tolist()], self.tokens.canvas)
        positions = positions.numpy()
        for array in (sizes, positions, *units):
            array.setflags(write=False)
        return PreparedMiniCPMPrompt(tuple(ids), units, sizes, per_image, positions, int(delta), spans, hashes)

    def continued(self, prepared: PreparedMiniCPMPrompt, tokens: Sequence[int]) -> PreparedMiniCPMPrompt:
        """The same images, the prompt gone on past ``prepared``'s tokens (text only), positions extended."""

        tokens = tuple(int(t) for t in tokens)
        if tokens[:len(prepared.token_ids)] != prepared.token_ids:
            raise ValueError("a continued image prompt must extend the prepared prompt's tokens")
        positions, delta = canvas_positions(tokens, [tuple(s) for s in prepared.target_sizes.tolist()],
                                            self.tokens.canvas)
        positions = positions.numpy()
        if not np.array_equal(positions[:, :len(prepared.token_ids)], prepared.position_ids) or \
                tuple(image_bounds(tokens, self.tokens.canvas)) != prepared.image_spans:
            raise ValueError("a continued image prompt may add text only")
        positions.setflags(write=False)
        return replace(prepared, token_ids=tokens, position_ids=positions, rope_delta=int(delta))


__all__ = ["MiniCPMImageProcessor", "MiniCPMPreprocessor", "PreparedMiniCPMPrompt", "Tokens", "canvas_positions",
           "compute_llm_grid", "image_bounds", "placeholder"]
