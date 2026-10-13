"""MiniCPM-V 4.7: found by model_type, CUDA only, Qwen3.6's MTP layer looked up in order, vision options checked."""

import json
from types import SimpleNamespace as NS

import pytest

from tensorfold import families
from tensorfold.families import minicpmv4_7

IDS = {"image_token_id": 248056, "image_start_id": 248078, "image_end_id": 248079, "slice_start_id": 248088,
       "slice_end_id": 248089, "newline_id": 198}


def _config(tmp_path, bits=4, group=64, **extra):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "minicpmv4_7", "quantization": {"bits": bits, "group_size": group, "mode": "affine"},
        "text_config": {"model_type": "qwen3_5_moe_text", "hidden_size": 2048},
        "vision_config": {"model_type": "minicpmv4_7_vision", "hidden_size": 1152}, **IDS, **extra}))
    return tmp_path


def test_the_family_is_found_by_its_top_level_model_type(tmp_path):
    family = families.detect(_config(tmp_path))
    assert family.module == "tensorfold.families.minicpmv4_7"
    assert family.title == "MiniCPM-V 4.7"
    assert families.backends_of(family) == ("cuda",)


def test_macs_are_refused_with_the_cuda_only_message(tmp_path):
    from tensorfold.cli import _backend

    family = families.detect(_config(tmp_path))
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        _backend("mlx", family)
    assert _backend("cuda", family) == "cuda"


def test_only_4_bit_groups_of_64_are_read(tmp_path):
    minicpmv4_7.check(_config(tmp_path))
    with pytest.raises(ValueError, match="groups of 64"):
        minicpmv4_7.check(_config(tmp_path, bits=8))
    families.require_readable(families.detect(_config(tmp_path)), json.loads((tmp_path / "config.json").read_text()),
                              "cuda")


@pytest.mark.parametrize("name,skipped", [
    ("model.vision_tower.encoder.layers.0.mlp.fc1.weight", True),
    ("model.merger.mlp.0.linear_1.weight", True),
    ("mtp.layers.0.self_attn.q_proj.weight", True),
    ("language_model.mtp.fc.weight", True),
    ("language_model.model.layers.0.linear_attn.in_proj_qkv.weight", False),
    ("language_model.lm_head.weight", False),
    ("language_model.model.embed_tokens.weight", False),
])
def test_the_text_loader_leaves_the_tower_merger_and_mtp_alone(name, skipped):
    assert minicpmv4_7.text_skip(name) is skipped


def test_mtp_lookup_order_env_then_side_file_then_cached_repo(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    model = _config(tmp_path / "model")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "mtp-4bit.safetensors").write_bytes(b"x")
    cached = lambda repo: cache if repo == minicpmv4_7.MTP_REPO else None        # noqa: E731
    monkeypatch.delenv("TF_QWEN36_MTP", raising=False)
    path, source = minicpmv4_7.mtp_file(model, cached=cached)
    assert path == cache / "mtp-4bit.safetensors" and "cached" in source
    (model / "mtp-4bit.safetensors").write_bytes(b"y")
    assert minicpmv4_7.mtp_file(model, cached=cached) == (model / "mtp-4bit.safetensors", "the checkpoint's side file")
    named = tmp_path / "named.safetensors"
    named.write_bytes(b"z")
    monkeypatch.setenv("TF_QWEN36_MTP", str(named))
    assert minicpmv4_7.mtp_file(model, cached=cached) == (named, "TF_QWEN36_MTP")
    monkeypatch.setenv("TF_QWEN36_MTP", "0")
    assert minicpmv4_7.mtp_file(model, cached=cached)[0] is None
    monkeypatch.setenv("TF_QWEN36_MTP", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        minicpmv4_7.mtp_file(model, cached=cached)


def test_no_mtp_layer_is_refused_unless_drafts_are_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_QWEN36_MTP", "0")
    with pytest.raises(ValueError, match="no MTP layer"):
        minicpmv4_7.cuda_engine(_config(tmp_path))
    with pytest.raises(ValueError, match="one GPU"):
        minicpmv4_7.cuda_engine(_config(tmp_path), tp=2)
    with pytest.raises(ValueError, match="draft model"):
        minicpmv4_7.cuda_engine(_config(tmp_path), drafter="x")


def test_vision_config_needs_the_canvas_token_ids(tmp_path):
    from tensorfold.vision.config import validate_vision_config

    config = json.loads((_config(tmp_path)).joinpath("config.json").read_text())
    assert validate_vision_config(config, "minicpmv4_7")["hidden_size"] == 1152
    del config["slice_end_id"]
    with pytest.raises(ValueError, match="image-token"):
        validate_vision_config(config, "minicpmv4_7")


def test_vision_with_parallel_streams_is_refused_before_loading(tmp_path):
    from tensorfold.serve_options import check

    family = families.detect(_config(tmp_path))
    args = NS(vision=True, parallel="2", vision_urls=False, vision_max_images=None, vision_image_tokens=None,
              vision_offload=False, kv_dtype="bf16", decode_share=None, checkpoint_slots=None, prefill_fp8=None,
              name_priority=None, alias=None)
    with pytest.raises(ValueError, match="one stream"):
        check(args, family, "cuda", tmp_path)
    with pytest.raises(ValueError, match="one stream"):
        minicpmv4_7.cuda_engine(tmp_path, vision=True, parallel=2, no_drafts=True)
