"""The Mojo tree attention (attention_rocm.mojo, TF_ROCM_TREE_KERNEL=mojo) gives the HIP WMMA kernels' bits: every
chunk partial (o, m, l) of the shared and tail kernels, and every merged row."""

import random

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import gfx12, hip  # noqa: E402

if not hip() or not gfx12():
    pytest.skip("the WMMA tree attention is gfx12's", allow_module_level=True)

from tensorfold.cuda import mojo  # noqa: E402

try:
    mojo.mojo_binary()
except RuntimeError:
    pytest.skip("no Mojo compiler", allow_module_level=True)

from tensorfold.cuda.kernels import attention as shared  # noqa: E402

H27, HK27, D = 24, 4, 256          # the 27B's attention: 24 query heads, 4 KV heads, head size 256
SENTINEL = 12345.678               # partials no kernel writes keep it in both runs


def _inputs(w, p, *, h=H27, hk=HK27, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(4100 + w + p + seed)
    q = torch.randn((w, h, D), generator=gen, device="cuda").bfloat16()
    kn = torch.randn((w, hk, D), generator=gen, device="cuda").bfloat16()
    vn = torch.randn((w, hk, D), generator=gen, device="cuda").bfloat16()
    kc = torch.randn((p + 5, hk, D), generator=gen, device="cuda").bfloat16()
    vc = torch.randn((p + 5, hk, D), generator=gen, device="cuda").bfloat16()
    return q, kn, vn, kc, vc


def _fp8(inputs):
    """test_attention's FP8 inputs: nodes rounded to FP8 values, caches as packed rows."""

    from tensorfold.cuda.kernels import kv8

    q, kn, vn, kc, vc = inputs
    kn, vn = (kv8.unpack(kv8.reference_pack(t)) for t in (kn, vn))
    return q, kn, vn, kv8.reference_pack(kc), kv8.reference_pack(vc)


def _tree(w):
    return [-1] + [(i - 1) // 2 for i in range(1, w)]


def _offs(caches):
    return torch.tensor(shared.offsets(caches, "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)


def _partials(kind, q, kn, vn, offs, plan, scale, kv8, launch):
    w, h, _ = q.shape
    hk = kn.shape[1]
    ext = shared._mojo() if kind == "mojo" else shared._rocm()
    po = torch.full((plan.chunks, w, h, D), SENTINEL, device="cuda")
    pm = torch.full((plan.chunks, w, h), SENTINEL, device="cuda")
    pl = torch.full((plan.chunks, w, h), SENTINEL, device="cuda")
    cw, pipe = launch or shared.rocm_launch(-(-w * (h // hk) // shared.QUERY_TILE), plan.chunks)
    origin = shared.base(q.device)
    ext.shared(q, origin, offs, plan.streams, plan.items, po, pm, pl, hk, cw, pipe, scale, kv8)
    ext.tail(q, kn, vn, origin, offs, plan.streams, plan.rows, plan.paths, plan.depths, po, pm, pl,
             1 + -(-shared.MAX_NODES // shared.CHUNK), scale, kv8)
    return [x.view(torch.int32) for x in (po, pm, pl)]


def _attend(monkeypatch, kind, q, kn, vn, offs, plan, scale, kv8):
    monkeypatch.setenv("TF_ROCM_TREE_KERNEL", kind)
    shared._rocm_kernel.cache_clear()
    try:
        return shared.attention(q, kn, vn, offs, plan, scale=scale, kv8=kv8).view(torch.int16)
    finally:
        shared._rocm_kernel.cache_clear()


def _same(monkeypatch, inputs, trees, lengths, scale=1 / 16, kv8=False, launch=None, caches=None):
    """HIP and Mojo partials and rows of one launch, bit for bit."""

    q, kn, vn = inputs[:3]
    plan = shared.plan(trees, lengths, q.shape[1] // kn.shape[1], "cuda")
    offs = _offs(caches if caches is not None else [inputs[3:]])
    for a, b, name in zip(_partials("wmma", q, kn, vn, offs, plan, scale, kv8, launch),
                          _partials("mojo", q, kn, vn, offs, plan, scale, kv8, launch), ("o", "m", "l")):
        assert torch.equal(a, b), f"partial {name} differs at {(a != b).nonzero()[:4].tolist()}"
    if launch is None:
        want = _attend(monkeypatch, "wmma", q, kn, vn, offs, plan, scale, kv8)
        got = _attend(monkeypatch, "mojo", q, kn, vn, offs, plan, scale, kv8)
        assert torch.equal(want, got), f"rows differ at {(want != got).nonzero()[:4].tolist()}"


# test_attention.py's head-size-256 cases (its d=128 chain runs Triton either way)
@pytest.mark.parametrize("kv8", [False, True])
@pytest.mark.parametrize("w,p", [(1, 0), (9, 13), (16, 511), (32, 512), (32, 1003), (128, 513), (7, 73),
                                 (12, 511), (128, 1300), (1, 3000), (12, 20501), (128, 20501)])
def test_mojo_gives_the_hip_bits_on_the_attention_test_trees(monkeypatch, w, p, kv8):
    inputs = _inputs(w, p)
    _same(monkeypatch, _fp8(inputs) if kv8 else inputs, [_tree(w)], [p], kv8=kv8)
    chain = [-1] + list(range(w - 1))
    _same(monkeypatch, _fp8(inputs) if kv8 else inputs, [chain], [p], scale=1 / 16, kv8=kv8)


@pytest.mark.parametrize("kv8", [False, True])
@pytest.mark.parametrize("w,p", [(12, 20501), (128, 1300), (1, 3000)])
def test_mojo_gives_the_hip_bits_with_any_compute_waves(monkeypatch, w, p, kv8):
    """test_compute_waves_never_change_the_bits' launches: 1, 2, 5 or 8 compute waves, with loader waves or without,
    alone and beside another stream."""

    inputs = _inputs(w, p)
    other = _inputs(3, 700, seed=1)
    if kv8:
        inputs, other = _fp8(inputs), _fp8(other)
    both = [torch.cat((inputs[j], other[j])) for j in range(3)]
    for cw in (1, 2, 5, 8):
        for pipe in (False, True):
            _same(monkeypatch, inputs, [[-1] + list(range(w - 1))], [p], scale=1 / 16, kv8=kv8, launch=(cw, pipe))
            _same(monkeypatch, both, [[-1] + list(range(w - 1)), [-1, 0, 1]], [p, 700], scale=1 / 16, kv8=kv8,
                  launch=(cw, pipe), caches=[inputs[3:], other[3:]])


@pytest.mark.parametrize("kv8", [False, True])
def test_mojo_gives_the_hip_bits_for_streams_in_one_launch(monkeypatch, kv8):
    rng = random.Random(5 + kv8)
    shapes = [(rng.randint(1, 16), rng.choice([0, 1, 100, 512, 513, 1300, 2600])) for _ in range(9)]
    parts = [_inputs(w, p, seed=i) for i, (w, p) in enumerate(shapes)]
    if kv8:
        parts = [_fp8(x) for x in parts]
    trees = [[-1] + [rng.randint(max(0, i - 4), i - 1) for i in range(1, w)] for w, _ in shapes]
    q, kn, vn = (torch.cat([x[j] for x in parts]) for j in range(3))
    _same(monkeypatch, (q, kn, vn), trees, [p for _, p in shapes], kv8=kv8, caches=[x[3:] for x in parts])


@pytest.mark.parametrize("w,p,context", [(1, 0, 4096), (4, 1300, 4096), (3, 2047, 2048), (4, 4000, 8192)])
def test_mojo_gives_the_hip_bits_on_a_padded_plan(monkeypatch, w, p, context):
    q, kn, vn, kc, vc = _inputs(w, p)
    parents = list(range(-1, w - 1))
    flat, items, chunks = shared.padded_host(parents, context, q.shape[1] // kn.shape[1])
    flat[w + 2], flat[w + 3] = p, -(-(p + w) // shared.CHUNK)
    plan = shared.from_packed(torch.tensor(flat, dtype=torch.int32, device="cuda"), 1, w, items, chunks)
    offs = _offs([(kc, vc)])
    for a, b in zip(_partials("wmma", q, kn, vn, offs, plan, 1 / 16, False, None),
                    _partials("mojo", q, kn, vn, offs, plan, 1 / 16, False, None)):
        assert torch.equal(a, b)
    assert torch.equal(_attend(monkeypatch, "wmma", q, kn, vn, offs, plan, 1 / 16, False),
                       _attend(monkeypatch, "mojo", q, kn, vn, offs, plan, 1 / 16, False))


def test_mojo_gives_the_hip_bits_on_strided_node_values(monkeypatch):
    w, p = 9, 513
    q, kn, vn, kc, vc = _inputs(w, p)
    kv = torch.zeros((w, 2 * HK27 * D), device="cuda", dtype=torch.bfloat16)
    kv[:, HK27 * D:] = vn.reshape(w, HK27 * D)
    view = kv[:, HK27 * D:].reshape(w, HK27, D)
    _same(monkeypatch, (q, kn, view, kc, vc), [_tree(w)], [p])


# The 27B's geometry (heads, KV heads and head size from qwen3_5's config) at about 1k, 16k and 64k committed keys,
# 1-16 verify rows, bf16 and packed FP8 caches; a chain (a drafted round's path) at each.
@pytest.mark.parametrize("kv8", [False, True])
@pytest.mark.parametrize("p", [1013, 16371, 64007])
def test_mojo_gives_the_hip_bits_on_the_27b_geometry(monkeypatch, p, kv8):
    base = _inputs(16, p, seed=27)
    if kv8:
        base = _fp8(base)
    for w in range(1, 17):
        rows = tuple(x[:w].contiguous() for x in base[:3])
        _same(monkeypatch, (*rows, *base[3:]), [[-1] + list(range(w - 1))], [p], scale=1 / 16, kv8=kv8)
