#!/usr/bin/env python3
"""MiniCPM-V 4.7 (BF16 HF checkpoint) -> MLX affine 4-bit checkpoint in the layout TensorFold's ``qwen3_5_moe`` reads.

MiniCPM-V 4.7's language model is Qwen3.6-35B-A3B (``qwen3_5_moe_text``) with a 248144-id vocabulary, so the text
weights are converted by mlx-lm's own ``qwen3_5_moe`` rules (fused experts split into ``switch_mlp.{gate,up,down}_proj``,
zero-centred norms shifted by one, conv1d moved to MLX layout, 4-bit groups of 64, router and shared-expert gate at
8 bits). Nothing here re-implements the quantizer: ``mlx_lm.convert`` runs on a text-only *view* of the checkpoint.

Steps:
  1. view    a folder of symlinks to the BF16 shards plus a Qwen3.6-style ``config.json`` (the text config under
             ``model_type: qwen3_5_moe``) and the tokenizer files, so mlx-lm treats it as Qwen3.6.
  2. convert ``mlx_lm.convert`` on the view. The view's shards also hold ``model.vision_tower.*`` and ``model.merger.*``,
             which mlx-lm's sanitize does not know, so they are dropped there (a wrapper around ``Model.sanitize``).
  3. finish  the vision tower and merger go into ``vision-bf16.safetensors`` under their original names (bytes
             copied through ``mx.save_safetensors``), the index is merged, ``config.json`` is MiniCPM's own plus the
             ``quantization`` / ``quantization_config`` block mlx-lm wrote, and the tokenizer/processor files are copied.

Output tensors: ``language_model.model.*`` and ``language_model.lm_head.*`` (the mlx-community Qwen3.6 names),
``model.vision_tower.*`` and ``model.merger.*`` (BF16, as in the source).

Run with the Python that has mlx-lm (e.g. ``~/.local/share/uv/tools/mlx-lm/bin/python``):
  python tools/minicpmv47_to_mlx.py --src ~/models/minicpm-v47-bf16 --out ~/models/MiniCPM-V-4.7-35B-A3B-MLX-4bit
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

VISION_PREFIXES = ("model.vision_tower.", "model.merger.")
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
              "processor_config.json", "preprocessor_config.json")
VISION_FILE = "vision-bf16.safetensors"


def build_view(src: Path, view: Path) -> None:
    """Symlink the shards, write the Qwen3.6-style config, link the tokenizer files."""

    view.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    text = dict(cfg["text_config"])
    text.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
    text["model_type"] = "qwen3_5_moe_text"
    text.setdefault("dtype", cfg.get("dtype", "bfloat16"))
    (view / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe", "text_config": text},
        indent=2))
    for p in sorted(src.glob("model-*.safetensors")) + [src / "model.safetensors.index.json"]:
        link = view / p.name
        if not link.exists():
            link.symlink_to(p.resolve())
    for name in COPY_FILES:
        if (src / name).is_file() and not (view / name).exists():
            (view / name).symlink_to((src / name).resolve())


def convert_text(view: Path, stage: Path) -> None:
    from mlx_lm.convert import convert
    from mlx_lm.models import qwen3_5_moe

    sanitize = qwen3_5_moe.Model.sanitize

    def text_only(self, weights):
        return sanitize(self, {k: v for k, v in weights.items() if not k.startswith(VISION_PREFIXES)})

    qwen3_5_moe.Model.sanitize = text_only
    # dtype: the view's text config carries ``dtype`` but pass it anyway; norms/A_log follow the model's cast_predicate
    convert(str(view), str(stage), quantize=True, q_group_size=64, q_bits=4, dtype="bfloat16")


def finish(src: Path, stage: Path, out: Path) -> None:
    import mlx.core as mx

    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(stage.glob("model*.safetensors")):
        shutil.move(str(p), out / p.name)

    src_index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(k for k in src_index if k.startswith(VISION_PREFIXES))
    by_file: dict[str, list[str]] = {}
    for k in names:
        by_file.setdefault(src_index[k], []).append(k)
    tensors = {}
    for f, keys in by_file.items():
        shard = mx.load(str(src / f))
        for k in keys:
            tensors[k] = shard[k]
    mx.eval(list(tensors.values()))
    mx.save_safetensors(str(out / VISION_FILE), tensors, metadata={"format": "mlx"})
    vision_bytes = sum(v.nbytes for v in tensors.values())

    index = json.loads((stage / "model.safetensors.index.json").read_text())
    index["weight_map"].update({k: VISION_FILE for k in tensors})
    index["metadata"]["total_size"] = int(index["metadata"].get("total_size", 0)) + vision_bytes
    index["weight_map"] = dict(sorted(index["weight_map"].items()))
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    staged = json.loads((stage / "config.json").read_text())
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization"] = staged["quantization"]
    cfg["quantization_config"] = staged["quantization_config"]
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)
    print(f"[INFO] wrote {out} ({len(tensors)} vision/merger tensors, {vision_bytes / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="BF16 MiniCPM-V 4.7 folder")
    ap.add_argument("--out", required=True, type=Path, help="output folder (must not exist)")
    ap.add_argument("--work", type=Path, help="scratch folder for the text view and mlx-lm's output (default: <out>.work)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    work = a.work or a.out.with_name(a.out.name + ".work")
    view, stage = work / "view", work / "stage"
    build_view(a.src, view)
    convert_text(view, stage)
    finish(a.src, stage, a.out)
    print(f"[INFO] mlx-lm's text-view config (for text-only mlx-lm runs) stays in {stage / 'config.json'}")


if __name__ == "__main__":
    main()
