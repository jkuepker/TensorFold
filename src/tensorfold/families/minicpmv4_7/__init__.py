"""MiniCPM-V 4.7 (minicpmv4_7): Qwen3.6 MoE's language model under a SigLIP tower, on the Qwen3.6 MoE CUDA engine."""
# The language model is a fine-tune of Qwen3.6-35B-A3B (``text_config.model_type`` qwen3_5_moe_text, vocabulary
# 248144) and ships no MTP layer; Qwen3.6's MTP side file drafts it (its rows hold no embedding or head, so the
# vocabulary does not matter), verified exactly by the target as on Qwen3.6. ``--vision`` adds MiniCPM's own tower and
# merger (BF16, plain torch, ``tensorfold.vision.minicpm_cuda``) with its canvas 3-D rotary positions. CUDA only.

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("minicpmv4_7",)
TITLE = "MiniCPM-V 4.7"
# converted locally with tools/minicpmv47_to_mlx.py (MLX 4-bit in groups of 64, routers 8-bit, vision-bf16.safetensors)
MODELS: tuple[str, ...] = ()
MTP_REPO = "TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP"       # Qwen3.6's MTP layer (mtp-4bit.safetensors)
MTP_FILE = "mtp-4bit.safetensors"
DRAFTER = ""                  # drafts with Qwen3.6's MTP layer
CUDA_QUANTIZATION = (4, 64)   # the Qwen3.6 MoE CUDA kernels read MLX affine 4-bit weights in groups of 64
CUDA_PREFILL_FP8 = True       # --prefill-fp8 as on Qwen3.6 MoE
VISION_PREFIXES = ("model.vision_tower.", "model.merger.")    # the tower and merger the text loader leaves alone


def check(model_dir: str | Path) -> None:
    """One GPU, MLX 4-bit weights in groups of 64."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quantization, read_config

    if quantization(read_config(model_dir)) != CUDA_QUANTIZATION:
        raise ValueError(f"{TITLE}'s CUDA engine reads MLX 4-bit weights in groups of 64 (tools/minicpmv47_to_mlx.py "
                         f"writes them); this checkpoint has {describe_quantization(read_config(model_dir))}. "
                         f"{OWN_MODEL_HELP}")


def vision_tensor(name: str) -> bool:
    return name.startswith(VISION_PREFIXES)


def text_skip(name: str) -> bool:
    """Tensors the text model does not read: the vision tower and merger, and any MTP layer (read on its own)."""

    return vision_tensor(name) or name.startswith("vision_tower") or ".mtp." in name or name.startswith("mtp.")


def mtp_file(model_dir: str | Path, *, cached=None) -> tuple[Path | None, str]:
    """Qwen3.6's MTP layer and where it came from: TF_QWEN36_MTP (a path, or 0 for none), a side file in the
    checkpoint, else the tested Qwen3.6 repo's cached file (``cached``: the hub lookup, for tests)."""

    import os

    choice = os.environ.get("TF_QWEN36_MTP", "")
    if choice == "0":
        return None, "TF_QWEN36_MTP=0"
    if choice:
        named = Path(choice).expanduser()
        if not named.is_file():
            raise FileNotFoundError(f"TF_QWEN36_MTP names {named}, which does not exist")
        return named, "TF_QWEN36_MTP"
    own = Path(model_dir) / MTP_FILE
    if own.is_file():
        return own, "the checkpoint's side file"
    if cached is None:
        from tensorfold import hub

        cached = hub.cached
    tested = cached(MTP_REPO)
    if tested is not None and (Path(tested) / MTP_FILE).is_file():
        return Path(tested) / MTP_FILE, f"{MTP_REPO}'s cached file"
    return None, "none found"


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, no_drafts: bool = False,
                mtp_drafts: int | None = None, context: int | None = None, vision: bool = False,
                vision_urls: bool = False, vision_offload: bool = False, **options: Any):
    """One GPU: the Qwen3.6 MoE engine on ``text_config``, Qwen3.6's MTP layer drafting; ``vision`` adds the tower."""

    if drafter:
        raise ValueError(f"{TITLE} drafts with Qwen3.6's MTP layer on CUDA: a separate draft model does not apply")
    if int(tp) != 1:
        raise ValueError(f"{TITLE} runs on one GPU: drop --tp")
    from tensorfold.families.qwen3_5_moe.cuda import DEPTH
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    streams = max(1, int(options.get("parallel") or 1))
    if streams > 1 and not 0 <= depth <= 15:
        raise ValueError(f"--parallel verifies up to 16 rows a stream: --mtp-drafts 0 to 15, not {depth}")
    if vision and streams > 1:
        raise ValueError(f"{TITLE} image input decodes one stream at a time: drop --parallel, or --vision")
    path = None
    if depth:
        path, source = mtp_file(model_dir)
        if path is None:
            raise ValueError(f"{TITLE} ships no MTP layer and none was found ({source}): fetch Qwen3.6's (0.5 GB, "
                             f"hf download {MTP_REPO} {MTP_FILE}), name it with TF_QWEN36_MTP=/path/{MTP_FILE}, or "
                             "pass --no-drafts")
        print(f"[tensorfold] {TITLE}: drafting with Qwen3.6's MTP layer, {path} ({source})", flush=True)
    from tensorfold.vision.minicpm_cuda import Lane

    # without --vision the lane only leaves the tower's tensors out of the memory estimate
    lane = Lane(Path(model_dir), enabled=vision, allow_urls=vision_urls, offload=vision_offload)
    return Qwen36Engine(Path(model_dir), depth=depth, context=context, context_explicit=options.get("context_explicit"),
                        streams=streams, mtp_path=path, skip=text_skip, vision=lane)
