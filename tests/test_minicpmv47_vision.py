"""MiniCPM-V 4.7 images on CUDA, checked on the CPU against transformers 5.19's MiniCPM-V 4.7 / 4.6 reference code:
canvas 3-D positions, slicing and patch packing, placeholder layout, and the ported tower and merger."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from tensorfold.vision import minicpm_processing as mp  # noqa: E402

IDS = {"image_token_id": 7, "video_token_id": 8, "image_start_id": 3, "image_end_id": 4, "slice_start_id": 5,
       "slice_end_id": 6, "newline_id": 2}
TOKENS = mp.Tokens(7, 3, 4, 5, 6, 2)
TEXT, NL, IM, IM_END, SL, SL_END, PAD = 1, 2, 3, 4, 5, 6, 7


def _image(thumb, slices=(), rows=0, index=None):
    """Token ids of one image as the processor lays it out: thumbnail pads, then rows of slices joined by "\\n"."""

    out = ([9, 10, 11] if index is not None else []) + [IM] + [PAD] * (thumb[0] * thumb[1] // 16) + [IM_END]
    if slices:
        cols = len(slices) // rows
        for r in range(rows):
            if r:
                out.append(NL)
            for h, w in slices[r * cols:(r + 1) * cols]:
                out += [SL] + [PAD] * (h * w // 16) + [SL_END]
    return out


def _reference():
    """transformers' get_rope_index on a model shell with only the config it reads."""

    modeling = pytest.importorskip("transformers.models.minicpmv4_7.modeling_minicpmv4_7")
    from types import SimpleNamespace

    shell = object.__new__(modeling.MiniCPMV4_7Model)
    torch.nn.Module.__init__(shell)
    shell.config = SimpleNamespace(downsample_mode="16x", **IDS)

    def positions(ids, sizes):
        ids_t = torch.tensor([ids])
        kinds = torch.tensor([[1 if t == PAD else 0 for t in ids]])
        pos, delta = modeling.MiniCPMV4_7Model.get_rope_index(shell, ids_t, mm_token_type_ids=kinds,
                                                              target_sizes=torch.tensor(sizes))
        return pos[:, 0, :], int(delta[0, 0])
    return positions


THUMB, SLICE = (24, 40), (32, 32)
CASES = {
    "one image, no slices, text around": ([TEXT] * 5 + _image(THUMB) + [NL, TEXT, TEXT], [THUMB]),
    "a 1 x 3 row of slices": ([TEXT] * 3 + _image(THUMB, [SLICE] * 3, rows=1) + [NL] + [TEXT] * 4,
                              [THUMB] + [SLICE] * 3),
    "2 x 3 slices with the image id": ([TEXT] * 4 + _image(THUMB, [(28, 36)] * 6, rows=2, index=0) + [NL, TEXT],
                                      [THUMB] + [(28, 36)] * 6),
    "two images with text between": ([TEXT] + _image((16, 20)) + [NL] + [TEXT] * 6 + _image(THUMB, [SLICE] * 2, rows=2)
                                     + [NL, TEXT], [(16, 20), THUMB, SLICE, SLICE]),
    "two images with only a newline between": ([TEXT] * 2 + _image((16, 20)) + [NL] + _image((20, 16)) + [NL, TEXT],
                                               [(16, 20), (20, 16)]),
    "an image ending the prompt": ([TEXT] * 2 + _image(THUMB, [SLICE] * 4, rows=2), [THUMB] + [SLICE] * 4),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_canvas_positions_match_transformers_get_rope_index(name):
    ids, sizes = CASES[name]
    ours, delta = mp.canvas_positions(ids, sizes, TOKENS.canvas)
    theirs, their_delta = _reference()(ids, sizes)
    assert torch.equal(ours, theirs), name
    assert delta == their_delta == int(ours.max()) + 1 - len(ids)


def test_the_canvas_folds_an_image_into_fewer_positions_than_tokens():
    ids, sizes = CASES["2 x 3 slices with the image id"]
    pos, delta = mp.canvas_positions(ids, sizes, TOKENS.canvas)
    assert delta < 0                                     # decoded rows sit closer than their index
    assert pos[:, -1].tolist() == [len(ids) - 1 + delta] * 3   # text after the canvas runs on one axis
    assert torch.equal(mp.canvas_positions([TEXT] * 4, [], TOKENS.canvas)[0], torch.arange(4).expand(3, -1))


@pytest.mark.parametrize("size", [(100, 80), (448, 448), (1280, 800), (800, 1280), (3000, 500), (640, 480)])
def test_slicing_and_packing_match_the_transformers_image_processor(size):
    from PIL import Image

    module = pytest.importorskip("transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6")
    rng = np.random.default_rng(sum(size))
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    reference = module.MiniCPMV4_6ImageProcessorPil(image_mean=[0.5] * 3, image_std=[0.5] * 3)(
        [image], return_tensors="np")
    units, sizes, grid = mp.MiniCPMImageProcessor().units(image)
    assert [list(s) for s in sizes] == reference["target_sizes"].tolist()
    assert list(grid) == list(reference["grids"][0])
    ours = np.concatenate(units, axis=-1)[None]
    assert ours.shape == reference["pixel_values"].shape
    np.testing.assert_allclose(ours, reference["pixel_values"], atol=1e-6)


def test_placeholder_text_matches_the_processor_layout():
    sizes = [(24, 40), (32, 32), (32, 32), (32, 32), (32, 32)]
    text = mp.placeholder(mp.Tokens(0, 0, 0, 0, 0, 0), 1, sizes, (2, 2))
    block = "<slice>" + "<|image_pad|>" * 64 + "</slice>"
    assert text == ("<image_id>1</image_id><image>" + "<|image_pad|>" * 60 + "</image>" + block * 2 + "\n" + block * 2)
    assert mp.placeholder(mp.Tokens(0, 0, 0, 0, 0, 0), None, sizes[:1], (0, 0)) == \
        "<image>" + "<|image_pad|>" * 60 + "</image>"


def _tokenizer(path):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "x": 1, "\n": 2, "0": 9, "1": 12, "user": 13, "What": 14}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("\n|[0-9]|[A-Za-z]+"), behavior="isolated")
    from tokenizers import AddedToken

    specials = {"<image>": 3, "</image>": 4, "<slice>": 5, "</slice>": 6, "<|image_pad|>": 7, "<|video_pad|>": 8,
                "<image_id>": 10, "</image_id>": 11}
    for text in sorted(specials, key=specials.get):      # added in id order, after the WordLevel vocab's own
        tokenizer.add_special_tokens([AddedToken(text, special=True)])
    data = json.loads(tokenizer.to_str())
    for entry in data["added_tokens"]:
        entry["id"] = specials[entry["content"]]
    data["model"]["vocab"].update(specials)
    (path / "tokenizer.json").write_text(json.dumps(data))


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "minicpmv4_7", "mrope_mode": "canvas", "downsample_mode": "16x", **IDS,
        "vision_config": {"hidden_size": 32}, "text_config": {"hidden_size": 16}}))
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": {
        "max_slice_nums": 9, "scale_resolution": 448, "patch_size": 14, "slice_mode": True, "use_image_id": True,
        "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "resample": 3, "downsample_mode": "16x"}}))
    _tokenizer(tmp_path)
    return tmp_path


class _Image:
    def __init__(self, width, height, detail="auto", seed=0):
        from PIL import Image

        rng = np.random.default_rng(seed)
        self.image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        self.detail, self.content_hash = detail, f"h{seed}"

    def to_pil(self):
        return self.image


def test_prepare_expands_markers_and_matches_reference_positions(checkpoint):
    pre = mp.MiniCPMPreprocessor.from_directory(checkpoint)
    rendered = "user\n<|image_pad|>\nWhat\n<|image_pad|>\nx"
    prepared = pre.prepare(rendered, [_Image(1280, 800), _Image(300, 200, seed=1)])
    ids = list(prepared.token_ids)
    assert ids[:3] == [13, 2, 10] and ids[3] == 9          # user \n <image_id> 0
    assert [ids[s - 1] for s, _ in prepared.image_spans].count(3) == 2      # two thumbnails after <image>
    assert prepared.units_per_image[1] == 1 and prepared.units_per_image[0] > 1
    assert prepared.visual_tokens == sum(1 for t in ids if t == 7)
    theirs, delta = _reference()(ids, prepared.target_sizes.tolist())
    np.testing.assert_array_equal(prepared.position_ids, theirs.numpy())
    assert prepared.rope_delta == delta
    longer = pre.continued(prepared, ids + [14, 2])
    assert longer.position_ids.shape == (3, len(ids) + 2)
    np.testing.assert_array_equal(longer.position_ids[:, :len(ids)], prepared.position_ids)
    assert longer.rope_delta == delta
    with pytest.raises(ValueError):
        pre.continued(prepared, ids[:-1] + [14])


def test_prepare_shrinks_slices_to_the_budget_and_low_detail_keeps_the_thumbnail(checkpoint):
    pre = mp.MiniCPMPreprocessor.from_directory(checkpoint)
    full = pre.prepare("<|image_pad|>", [_Image(1280, 800)])
    small = pre.prepare("<|image_pad|>", [_Image(1280, 800)], max_visual_tokens=200)
    low = pre.prepare("<|image_pad|>", [_Image(1280, 800, detail="low")])
    assert small.visual_tokens <= 200 < full.visual_tokens
    assert low.units_per_image == (1,)
    with pytest.raises(ValueError, match="marker"):
        pre.prepare("no marker", [_Image(64, 64)])
    with pytest.raises(ValueError, match="video"):
        pre.prepare("<|image_pad|>", [_Image(64, 64)], videos=[object()])


def _tiny():
    raw = {"model_type": "minicpmv4_7", "insert_layer_id": 1, **IDS,
           "text_config": {"model_type": "qwen3_5_moe_text", "hidden_size": 24},
           "vision_config": {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 3,
                             "num_attention_heads": 4, "patch_size": 2, "image_size": 16, "num_channels": 3,
                             "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6}}
    return raw


def test_the_ported_tower_and_merger_match_transformers():
    modeling = pytest.importorskip("transformers.models.minicpmv4_7.modeling_minicpmv4_7")
    configuration = pytest.importorskip("transformers.models.minicpmv4_7.configuration_minicpmv4_7")
    from tensorfold.vision.minicpm_cuda import build

    raw = _tiny()
    torch.manual_seed(0)
    config = configuration.MiniCPMV4_7Config(
        vision_config=dict(raw["vision_config"]), insert_layer_id=1,
        text_config={"model_type": "qwen3_5_moe_text", "hidden_size": 24, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 16,
                     "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 8,
                     "shared_expert_intermediate_size": 8, "linear_num_key_heads": 1, "linear_num_value_heads": 1,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8, "layer_types": ["full_attention"]},
        **IDS)
    config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    reference = modeling.MiniCPMV4_7Model(config).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, 0.2)
    ours = build(raw, raw["vision_config"]).eval()
    state = {k.removeprefix("model."): v
             for k, v in reference.state_dict().items() if k.startswith(("vision_tower.", "merger."))}
    ours.load_state_dict(state, strict=True)
    sizes = [(8, 12), (8, 8), (4, 8)]
    pixels = torch.randn(1, 3, 2, sum(h * w for h, w in sizes) * 2)
    with torch.no_grad():
        theirs = torch.cat(reference.get_image_features(pixels, torch.tensor(sizes)).pooler_output)
        mine = ours(pixels, sizes)
    assert mine.shape == (sum(h * w // 16 for h, w in sizes), 24)
    torch.testing.assert_close(mine, theirs, atol=2e-5, rtol=1e-4)
