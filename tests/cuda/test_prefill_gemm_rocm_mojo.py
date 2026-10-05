"""The Mojo prompt GEMM (``prefill_rocm.mojo``, ``TF_ROCM_PREFILL_GEMM=mojo``) gives the Triton ``_gemm``'s bits: the
existing prompt-matmul shapes and the 27B's projections at the prompt chunk sizes the engine takes, every tile."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import gfx12, hip  # noqa: E402

if not (hip() and gfx12()):
    pytest.skip("the Mojo prompt GEMM serves ROCm on gfx12", allow_module_level=True)

from tensorfold.cuda.kernels import qmm_groups  # noqa: E402

TILES = list(qmm_groups.PREFILL_TILES)
# the 27B's prompt projections (N, K): attn q, attn [k|v], attn o / gdn out, gdn qkv, gdn [z|b|a], mlp gate (up),
# mlp down
SHAPES_27B = [(12288, 5120), (2048, 5120), (5120, 6144), (10240, 5120), (6240, 5120), (17408, 5120), (5120, 17408)]
# chunk rows: prefill.chunks of the R9700's 2560-row chunks (4k: 2 x 2048, 16k: 7 x 2340/2341, 30k: 12 x 2500), the
# chunk itself, an 80 GB card's 4096, and a short last piece
ROWS_27B = [2048, 2341, 2500, 2560, 4096, 37]


def _weight(n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    words = words.to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
    return qmm_groups.to_groups(words, scales, biases), gen


def _both(monkeypatch, x, g, n, f32, tile="128x128"):
    outs = []
    for kind in ("triton", "mojo"):
        monkeypatch.setenv("TF_ROCM_PREFILL_GEMM", kind)
        monkeypatch.setenv("TF_ROCM_PREFILL_TILE", tile)
        qmm_groups.reload_settings()
        assert qmm_groups.prefill_gemm() == kind
        outs.append(qmm_groups.prefill_matmul(x, *g, n, f32=f32))
    qmm_groups.reload_settings()
    return outs


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    yield
    monkeypatch.delenv("TF_ROCM_PREFILL_GEMM", raising=False)
    monkeypatch.delenv("TF_ROCM_PREFILL_TILE", raising=False)
    qmm_groups.reload_settings()


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n,k", [(1, 128), (256, 128), (384, 128), (48, 5120), (1024, 5120), (10240, 5120),
                                 (17408, 5120), (5120, 17408), (5120, 6144), (1000, 1024), (1024, 6144)])
def test_mojo_prompt_gemm_gives_tritons_bits_at_test_shapes(monkeypatch, n, k, f32, tile):
    g, gen = _weight(n, k, n + k)
    for rows in (200, 333, 1, 7, 16, 64, 128, 256):
        x = torch.randn(rows, k, generator=gen, device="cuda").bfloat16()
        ref, got = _both(monkeypatch, x, g, n, f32, tile)
        assert torch.equal(ref, got), (rows, n, k)


@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n,k", SHAPES_27B)
def test_mojo_prompt_gemm_gives_tritons_bits_at_27b_chunks(monkeypatch, n, k, f32):
    g, gen = _weight(n, k, 7 * n + k)
    for rows in ROWS_27B:
        x = (torch.randn(rows, k, generator=gen, device="cuda") * 0.5).bfloat16()
        for tile in TILES:
            ref, got = _both(monkeypatch, x, g, n, f32, tile)
            assert torch.equal(ref, got), (rows, n, k, tile)
