"""MiniCPM-V 4.7's image tower and merger on the CUDA engine: BF16 in plain torch, beside the Qwen3.6 MoE text model.

The modules port transformers 5.19's ``modeling_minicpmv4_7.py`` (Apache-2.0, OpenBMB and the HuggingFace team):
a SigLIP encoder on NaViT-packed patches with bucketed 2-D position embeddings, the window-attention merger after
encoder layer ``insert_layer_id`` (2 x 2 windows -> one token), and the 2 x 2 MLP merger into the language model's
width. Each unit (a thumbnail or a slice) attends only to itself, as transformers' ``cu_seqlens`` split does.
"""

from __future__ import annotations

import json
import math
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

MAX_PATCHES = 16384                  # one tower call's patches (whole units, about 16 slices); a request goes on in runs
WORKSPACE_BYTES = 1024**3            # activations of one call (measured peak well under this) on the resident tower
OFFLOAD_WORKSPACE_BYTES = int(2 * 1024**3)   # --vision-offload: the tower's visit to the GPU plus one call's activations
PREFIXES = ("model.vision_tower.", "model.merger.")


def _config(model_dir: Path) -> tuple[dict, dict]:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    vision = raw.get("vision_config")
    if raw.get("model_type") != "minicpmv4_7" or not isinstance(vision, dict):
        raise ValueError("MiniCPM-V CUDA vision reads a MiniCPM-V 4.7 checkpoint with its vision_config")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "patch_size",
                "image_size"):
        if not isinstance(vision.get(key), int) or isinstance(vision[key], bool) or vision[key] <= 0:
            raise ValueError(f"invalid MiniCPM-V vision configuration: {key}")
    if vision["hidden_size"] % vision["num_attention_heads"]:
        raise ValueError("MiniCPM-V vision heads must divide its width")
    if str(raw.get("downsample_mode", "16x")) != "16x" or int(raw.get("merger_times", 1)) != 1 or \
            list(raw.get("merge_kernel_size", (2, 2))) != [2, 2] or \
            list(vision.get("window_kernel_size", (2, 2))) != [2, 2]:
        raise ValueError("MiniCPM-V CUDA vision reads the 16x tower (2 x 2 window merger, one 2 x 2 merger)")
    insert = int(raw.get("insert_layer_id", 6))
    if not 0 <= insert < vision["num_hidden_layers"]:
        raise ValueError("MiniCPM-V's insert_layer_id is outside its encoder")
    return raw, vision


def build(raw: dict, vision: dict):
    """The tower and merger as torch modules named as the checkpoint names them (less ``model.``)."""

    import torch
    import torch.nn.functional as F
    from torch import nn

    D, I, heads = vision["hidden_size"], vision["intermediate_size"], vision["num_attention_heads"]
    eps = float(vision.get("layer_norm_eps", 1e-6))
    p = vision["patch_size"]
    side = vision["image_size"] // p
    hd = D // heads
    text = raw.get("text_config") or {}
    out = int(text.get("hidden_size") or raw["hidden_size"])
    insert = int(raw.get("insert_layer_id", 6))

    def gelu_tanh(x):
        return F.gelu(x, approximate="tanh")

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(D, D) for _ in range(4))

        def forward(self, x, groups: Sequence[int] | None = None, window: int = 0):
            """``groups``: the rows of each unit, attending only within it; ``window``: fixed groups of that many."""

            n = x.shape[0]
            q, k, v = (proj(x).view(n, heads, hd) for proj in (self.q_proj, self.k_proj, self.v_proj))
            if window:
                q, k, v = (t.view(n // window, window, heads, hd).transpose(1, 2) for t in (q, k, v))
                o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n, D)
            else:
                outs = []
                for qs, ks, vs in zip(q.split(groups), k.split(groups), v.split(groups)):
                    outs.append(F.scaled_dot_product_attention(qs.transpose(0, 1)[None], ks.transpose(0, 1)[None],
                                                               vs.transpose(0, 1)[None])[0].transpose(0, 1))
                o = torch.cat(outs).reshape(n, D)
            return self.out_proj(o)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(D, I), nn.Linear(I, D)

        def forward(self, x):
            return self.fc2(gelu_tanh(self.fc1(x)))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm1, self.layer_norm2 = nn.LayerNorm(D, eps=eps), nn.LayerNorm(D, eps=eps)
            self.self_attn, self.mlp = Attention(), MLP()

        def forward(self, x, groups):
            x = x + self.self_attn(self.layer_norm1(x), groups)
            return x + self.mlp(self.layer_norm2(x))

    class WindowMerger(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.layer_norm1 = Attention(), nn.LayerNorm(D, eps=eps)
            self.pre_norm = nn.LayerNorm(4 * D, eps=eps)
            self.linear_1, self.linear_2 = nn.Linear(4 * D, 4 * I), nn.Linear(4 * I, D)

        def forward(self, x, sizes):
            order, start = [], 0
            for h, w in sizes:                          # each 2 x 2 window's rows together, windows in raster order
                idx = torch.arange(h * w, device=x.device).view(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3)
                order.append(idx.reshape(-1) + start)
                start += h * w
            order = torch.cat(order)
            x = x[order]
            x = x + self.self_attn(self.layer_norm1(x), window=4)
            patch = x.view(-1, 4, D)
            y = self.linear_2(gelu_tanh(self.linear_1(self.pre_norm(patch.flatten(1)))))
            return y + patch.mean(dim=1)

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embedding = nn.Conv2d(3, D, kernel_size=p, stride=p)
            self.position_embedding = nn.Embedding(side * side, D)

        def forward(self, pixels, sizes):
            x = self.patch_embedding(pixels).flatten(2)[0].transpose(0, 1)      # [patches, D], raster per unit
            bounds = torch.arange(1 / side, 1.0, 1 / side, device=x.device)
            ids = []
            for h, w in sizes:
                bh = torch.bucketize(torch.arange(h, device=x.device) / h, bounds, right=True)
                bw = torch.bucketize(torch.arange(w, device=x.device) / w, bounds, right=True)
                ids.append((bh[:, None] * side + bw).flatten())
            return x + self.position_embedding(torch.cat(ids))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Layer() for _ in range(vision["num_hidden_layers"]))

    class VisionTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings, self.encoder = Embeddings(), Encoder()
            self.post_layernorm = nn.LayerNorm(D, eps=eps)
            self.vit_merger = WindowMerger()

        def forward(self, pixels, sizes):
            x = self.embeddings(pixels, sizes)
            groups = [h * w for h, w in sizes]
            for i, layer in enumerate(self.encoder.layers):
                x = layer(x, groups)
                if i == insert:
                    x = self.vit_merger(x, sizes)
                    sizes = [(h // 2, w // 2) for h, w in sizes]
                    groups = [h * w for h, w in sizes]
            return self.post_layernorm(x), sizes

    class DownsampleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_norm = nn.LayerNorm(4 * D, eps=1e-6)
            self.linear_1, self.linear_2 = nn.Linear(4 * D, 4 * D), nn.Linear(4 * D, out)

        def forward(self, x):
            return self.linear_2(F.gelu(self.linear_1(self.pre_norm(x))))

    class Merger(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.ModuleList([DownsampleMLP()])

        def forward(self, x, sizes):
            parts = []
            for unit, (h, w) in zip(x.split([h * w for h, w in sizes]), sizes):
                grouped = unit.view(h // 2, 2, w // 2, 2, D).permute(0, 2, 1, 3, 4).reshape(-1, 4 * D)
                parts.append(self.mlp[0](grouped))
            return torch.cat(parts)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower, self.merger = VisionTower(), Merger()

        def forward(self, pixels, sizes):
            """pixels [1, 3, patch, patches * patch] (units packed along the last axis), sizes [(h, w)] each unit's
            patch grid -> [sum(h * w) / 16, out] features in unit order."""

            if any(h % 4 or w % 4 for h, w in sizes):
                raise ValueError("MiniCPM-V units' patch grids must be multiples of 4")
            x, merged = self.vision_tower(pixels, [(int(h), int(w)) for h, w in sizes])
            return self.merger(x, merged)

    return Model()


def tower_tensors(model_dir: Path) -> dict[str, tuple[Path, dict]]:
    """The tower and merger tensors' files and headers (via the index, else every shard), by checkpoint name."""

    from tensorfold.cuda.capacity import headers

    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    where = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    found = {}
    files = sorted({model_dir / f for n, f in where.items() if n.startswith(PREFIXES)}) if where is not None else \
        sorted(model_dir.glob("*.safetensors"))
    for path in files:
        for name, info in headers(model_dir, files=[path]).items():
            if name.startswith(PREFIXES):
                found[name] = (path, info)
    return found


def checkpoint_tower(model_dir: Path) -> tuple[dict, dict, int]:
    """Check the tower's tensors against the configuration before any allocation: (config, vision config, bytes)."""

    import torch

    from tensorfold.cuda.capacity import SIZES

    raw, vision = _config(model_dir)
    with torch.device("meta"):
        expected = {"model." + k: list(v.shape) for k, v in build(raw, vision).state_dict().items()}
    found = tower_tensors(model_dir)
    if set(found) != set(expected):
        missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
        raise ValueError(f"the checkpoint's MiniCPM-V tower differs from its configuration (missing {missing[:3]}, "
                         f"unexpected {extra[:3]}); convert it with tools/minicpmv47_to_mlx.py")
    for name, (_, info) in found.items():
        if info.get("dtype") not in ("BF16", "F16", "F32") or list(info.get("shape", ())) != expected[name]:
            raise ValueError(f"MiniCPM-V vision tensor {name} has {info.get('dtype')} {info.get('shape')}, not a "
                             f"floating-point {expected[name]}")
    return raw, vision, sum(math.prod(info["shape"]) * max(2, SIZES[info["dtype"]]) for _, info in found.values())


class Lane:
    """The engine's image lane: the tower's bytes and workspace in admission, the tower loaded after the text model
    (``start``). Disabled (no --vision), the tower's tensors count nothing and nothing loads."""

    def __init__(self, model_dir: Path, *, enabled: bool = True, allow_urls: bool = False, offload: bool = False):
        self.model_dir, self.enabled, self.allow_urls, self.offload = Path(model_dir), bool(enabled), allow_urls, offload

    def geometry(self, base):
        if not self.enabled:
            return base

        def geometry(text):
            from tensorfold.cuda.capacity import Geometry

            result = base(text) if callable(base) else base
            checkpoint_tower(self.model_dir)
            reserve = OFFLOAD_WORKSPACE_BYTES if self.offload else WORKSPACE_BYTES
            return Geometry(lambda slots: result.bytes_at(slots) + reserve, result.reserve, result.minimum_slots)
        return geometry

    def transform(self, base):
        def transform(name, info):
            if name.startswith(PREFIXES):
                if not self.enabled or self.offload:
                    return 0, 0                         # not loaded, or resident in host RAM between images
                from tensorfold.cuda.capacity import SIZES

                return math.prod(info["shape"]) * max(2, SIZES[info["dtype"]]), 0
            return base(name, info)
        return transform

    def start(self, device):
        if not self.enabled:
            return None
        frontend = MiniCPMCudaVision(self.model_dir, device, allow_urls=self.allow_urls, offload=self.offload)
        frontend.warm()
        return frontend


class MiniCPMCudaVision:
    """The server's image frontend (``prepare``, ``continued``) and the engine's encoder (``encode``)."""

    videos = False

    def __init__(self, model_dir, device, allow_urls: bool = False, offload: bool = False):
        import torch
        from safetensors import safe_open

        from .minicpm_processing import MiniCPMPreprocessor

        self.allow_urls, self.offload, self.device = allow_urls, offload, device
        self._lock = threading.Lock()          # one image run on the GPU at a time when the tower is offloaded
        raw, vision, self.weight_bytes = checkpoint_tower(Path(model_dir))
        self.frontend = MiniCPMPreprocessor.from_directory(model_dir)
        self.config = self.frontend.config
        self.image_token = int(raw["image_token_id"])
        self.width = int((raw.get("text_config") or {}).get("hidden_size") or raw["hidden_size"])
        with torch.device("meta"):
            model = build(raw, vision)
        resident = "cpu" if offload else device
        tensors, by_file = {}, {}
        for name, (path, _) in tower_tensors(Path(model_dir)).items():
            by_file.setdefault(path, []).append(name)
        for path, names in by_file.items():
            with safe_open(str(path), framework="pt", device="cpu") as source:
                for name in names:
                    tensors[name[len("model."):]] = source.get_tensor(name).to(device=resident, dtype=torch.bfloat16)
        model.load_state_dict(tensors, strict=True, assign=True)
        self.tower = model.eval()

    def prepare(self, rendered_prompt, images, **kwargs):
        kwargs.pop("max_image_tokens", None)
        return self.frontend.prepare(rendered_prompt, images, **kwargs)

    def continued(self, prepared, tokens):
        return self.frontend.continued(prepared, tokens)

    @contextmanager
    def _on_gpu(self):
        if not self.offload:
            yield
            return
        import torch

        with self._lock:
            try:
                self.tower.to(self.device)
                yield
            finally:
                self.tower.to("cpu")
                torch.cuda.empty_cache()

    def warm(self) -> None:
        """Load the tower's kernels at startup on one small unit (16 x 16 patches)."""

        import torch

        p = int(self.tower.vision_tower.embeddings.patch_embedding.kernel_size[0])
        with self._on_gpu(), torch.inference_mode():
            self.tower(torch.zeros((1, 3, p, 256 * p), dtype=torch.bfloat16, device=self.device), [(16, 16)])
            torch.cuda.synchronize()

    def features(self, prepared) -> Any:
        """[visual tokens, width] bf16 features of every unit in prompt order, in runs of whole units."""

        import numpy as np
        import torch

        from .qwen_cuda import image_runs

        sizes = [tuple(int(v) for v in s) for s in prepared.target_sizes.tolist()]
        patches = [h * w for h, w in sizes]
        parts = []
        with self._on_gpu(), torch.inference_mode():
            for begin, end in image_runs(patches, MAX_PATCHES):
                packed = np.concatenate(prepared.pixel_values[begin:end], axis=-1)[None]     # a copy: writable
                pixels = torch.from_numpy(packed).to(device=self.device, dtype=torch.bfloat16)
                parts.append(self.tower(pixels, sizes[begin:end]).to(torch.bfloat16))
        return torch.cat(parts).contiguous()

    def encode(self, prepared, prompt):
        import torch

        from .qwen_cuda import EncodedVision, validate_encoded

        if tuple(prompt) != tuple(prepared.token_ids):
            raise ValueError("vision preparation belongs to different prompt tokens")
        rows = tuple(i for start, end in prepared.image_spans for i in range(start, end))
        positions = prepared.position_ids.tolist()
        validate_encoded(rows, positions, prepared.rope_delta, prompt, self.image_token,
                         (prepared.visual_tokens, self.width), self.width)
        features = self.features(prepared)
        if tuple(features.shape) != (len(rows), self.width):
            raise ValueError("the MiniCPM-V tower returned a different number of image features")
        return EncodedVision(rows, features, torch.tensor(positions, dtype=torch.int32, device=self.device),
                             prepared.rope_delta)


__all__ = ["Lane", "MiniCPMCudaVision", "build", "checkpoint_tower"]
