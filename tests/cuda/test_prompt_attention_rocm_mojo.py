"""The Mojo prompt attention (``attention_rocm.mojo``'s prompt kernels, ``TF_ROCM_ATTN_KERNEL=mojo``) gives
``attention_rocm.cu``'s prompt_kernel bits: every head grouping the HIP kernel takes, one or two row tiles a block,
with loader waves or without, bf16 and packed FP8 caches, and the 27B's prompt chunks at 4k, 16k and 30k."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import gfx12, hip  # noqa: E402

if not (hip() and gfx12()):
    pytest.skip("the Mojo prompt attention serves ROCm on gfx12", allow_module_level=True)

from tensorfold.cuda.kernels import attention as tree  # noqa: E402
from tensorfold.cuda.kernels import kv8, prefill_attention  # noqa: E402

D = 256
HEADS = [(24, 4), (8, 8), (16, 8), (16, 4), (32, 4), (16, 2), (32, 32)]
CHUNKS = [(0, 1), (0, 100), (0, 333), (17, 7), (64, 64), (1000, 129), (4133, 300)]
# prefill.chunks of the R9700's 2560-row chunks: 4k (2 x 2048), 16k (7 x 2340/2341), 30k (12 x 2500)
CHUNKS_27B = [(0, 2048), (2048, 2048), (0, 2340), (2340, 2341), (9362, 2341), (14043, 2341), (0, 2500),
              (15000, 2500), (27500, 2500)]


def _caches(gen, keys, hk, packed: bool):
    k = (torch.randn(keys + 50, hk, D, generator=gen, device="cuda") * 1.5).bfloat16()
    v = torch.randn(keys + 50, hk, D, generator=gen, device="cuda").bfloat16()
    k[keys:] = float("nan")                                    # past the chunk's keys: never read
    v[keys:] = float("nan")
    if packed:
        k, v = kv8.pack(k)[1], kv8.pack(v)[1]
    return k, v


def _both(q, k, v, p0, rb, pipe):
    scale = D ** -0.5
    ref, got = torch.empty_like(q), torch.empty_like(q)
    prefill_attention._rocm().attention(q, k, v, ref, p0, scale, rb, pipe)
    tree._mojo(q.get_device()).attention(q, k, v, got, p0, scale, rb, pipe)
    return ref, got


@pytest.mark.parametrize("packed", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("heads,kv_heads", HEADS)
def test_mojo_prompt_attention_gives_the_hip_bits(heads, kv_heads, packed):
    gen = torch.Generator(device="cuda").manual_seed(heads * 31 + kv_heads)
    for p0, w in CHUNKS + ([(70001, 257)] if heads == 24 else []):
        k, v = _caches(gen, p0 + w, kv_heads, packed)
        q = (torch.randn(w, heads, D, generator=gen, device="cuda") * 1.5).bfloat16()
        for rb in (1, 2):
            for pipe in (True, False):
                ref, got = _both(q, k, v, p0, rb, pipe)
                assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (p0, w, rb, pipe)


@pytest.mark.parametrize("packed", [False, True], ids=["bf16", "fp8"])
def test_mojo_prompt_attention_gives_the_hip_bits_at_27b_chunks(packed):
    gen = torch.Generator(device="cuda").manual_seed(27)
    for p0, w in CHUNKS_27B:
        k, v = _caches(gen, p0 + w, 4, packed)
        q = (torch.randn(w, 24, D, generator=gen, device="cuda") * 1.5).bfloat16()
        rb, pipe = prefill_attention.rocm_rows(w)
        ref, got = _both(q, k, v, p0, rb, pipe)
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (p0, w)


def test_mojo_prompt_attention_is_routed_by_tf_rocm_attn_kernel(monkeypatch):
    gen = torch.Generator(device="cuda").manual_seed(3)
    k, v = _caches(gen, 700, 4, False)
    q = torch.randn(300, 24, D, generator=gen, device="cuda").bfloat16()
    outs = []
    for kind in ("wmma", "mojo"):
        monkeypatch.setenv("TF_ROCM_ATTN_KERNEL", kind)
        prefill_attention._rocm_kernel.cache_clear()
        assert prefill_attention._rocm_kernel(24, 4, D) == kind
        outs.append(prefill_attention.attention(q, k, v, 400, scale=D ** -0.5))
    prefill_attention._rocm_kernel.cache_clear()
    assert torch.equal(outs[0], outs[1])


def test_mojo_prompt_attention_needs_no_hip_extension(monkeypatch):
    """TF_ROCM_ATTN_KERNEL=mojo routes and runs without attention_rocm.cu."""

    gen = torch.Generator(device="cuda").manual_seed(4)
    k, v = _caches(gen, 700, 4, False)
    q = torch.randn(300, 24, D, generator=gen, device="cuda").bfloat16()
    want = torch.empty_like(q)
    prefill_attention._rocm().attention(q, k, v, want, 400, D ** -0.5, *prefill_attention.rocm_rows(300))

    def refuse():
        raise AssertionError("the Mojo prompt attention built the HIP extension")

    monkeypatch.setenv("TF_ROCM_ATTN_KERNEL", "mojo")
    monkeypatch.setattr(prefill_attention, "_rocm", refuse)
    prefill_attention._rocm_kernel.cache_clear()
    try:
        got = prefill_attention.attention(q, k, v, 400, scale=D ** -0.5)
    finally:
        prefill_attention._rocm_kernel.cache_clear()
    assert torch.equal(want, got)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_mojo_prompt_attention_gives_the_hip_bits_on_each_gpu():
    """One process, two GPUs: each loads its own module, and a launch from the other GPU's context stays on its
    tensors' GPU."""

    for packed in (False, True):
        for dev in (0, 1, 0):
            with torch.cuda.device(dev):
                gen = torch.Generator(device="cuda").manual_seed(5)
                k, v = _caches(gen, 1129, 4, packed)
                q = (torch.randn(129, 24, D, generator=gen, device="cuda") * 1.5).bfloat16()
                ref, got = _both(q, k, v, 1000, 1, True)
                assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), (dev, packed)
        with torch.cuda.device(1):
            gen = torch.Generator(device="cuda").manual_seed(5)
            k, v = _caches(gen, 1129, 4, packed)
            q = (torch.randn(129, 24, D, generator=gen, device="cuda") * 1.5).bfloat16()
            ref = torch.empty_like(q)
            prefill_attention._rocm().attention(q, k, v, ref, 1000, D ** -0.5, 1, True)
        with torch.cuda.device(0):
            got = torch.empty_like(q)
            tree._mojo(q.get_device()).attention(q, k, v, got, 1000, D ** -0.5, 1, True)
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), packed
