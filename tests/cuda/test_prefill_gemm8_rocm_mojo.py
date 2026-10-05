"""The Mojo FP8 prompt GEMM (``prefill_rocm.mojo``'s gemm8 kernels, ``TF_ROCM_PREFILL_GEMM=mojo`` with --prefill-fp8)
gives the Triton ``_gemm8``'s bits: the existing FP8 prompt-matmul shapes and the 27B's projections at the prompt
chunk sizes the engine takes, every tile."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import gfx12, hip  # noqa: E402

if not (hip() and gfx12()):
    pytest.skip("the Mojo prompt GEMM serves ROCm on gfx12", allow_module_level=True)

from tensorfold.cuda.kernels import qmm_groups  # noqa: E402

TILES = list(qmm_groups.PREFILL8_TILES)
SHAPES_27B = [(12288, 5120), (2048, 5120), (5120, 6144), (10240, 5120), (6240, 5120), (17408, 5120), (5120, 17408)]
ROWS_27B = [2048, 2341, 2500, 2560, 4096, 37]


def _weight(n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    words = words.to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
    return qmm_groups.to_groups(words, scales, biases), gen


def _rows(m: int, k: int, gen):
    """``prefill_glue``'s FP8 inputs: e4m3 rows (scaled to 448 a row), bf16 group sums, fp32 row scales."""

    x = torch.randn(m, k, generator=gen, device="cuda")
    x[:, 5] *= 30                                                  # an outlier channel
    a = x.abs().amax(1).clamp_min(1e-30) / 448.0
    x8 = (x / a[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    xs = (x.view(m, k // 64, 64).sum(2) / a[:, None]).bfloat16()
    return x8, xs, a.contiguous()


def _both(monkeypatch, rows, g, n, f32, tile):
    outs = []
    for kind in ("triton", "mojo"):
        monkeypatch.setenv("TF_ROCM_PREFILL8_GEMM", kind)
        monkeypatch.setenv("TF_ROCM_PREFILL8_TILE", tile)
        qmm_groups.reload_settings()
        assert qmm_groups.prefill8_gemm() == kind
        outs.append(qmm_groups.prefill_matmul8(rows, *g, n, f32=f32))
    qmm_groups.reload_settings()
    return outs


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    yield
    monkeypatch.delenv("TF_ROCM_PREFILL8_GEMM", raising=False)
    monkeypatch.delenv("TF_ROCM_PREFILL8_TILE", raising=False)
    qmm_groups.reload_settings()


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n,k", [(1, 128), (256, 128), (384, 128), (48, 5120), (1024, 5120), (17408, 5120),
                                 (5120, 17408), (5120, 6144), (1024, 6144), (1000, 1024)])
def test_mojo_fp8_prompt_gemm_gives_tritons_bits_at_test_shapes(monkeypatch, n, k, f32, tile):
    g, gen = _weight(n, k, n + 3 * k)
    for m in (300, 1, 2, 5, 7, 64, 256):
        ref, got = _both(monkeypatch, _rows(m, k, gen), g, n, f32, tile)
        assert torch.equal(ref, got), (m, n, k)


@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n,k", SHAPES_27B)
def test_mojo_fp8_prompt_gemm_gives_tritons_bits_at_27b_chunks(monkeypatch, n, k, f32):
    g, gen = _weight(n, k, 5 * n + k)
    for m in ROWS_27B:
        rows = _rows(m, k, gen)
        for tile in TILES:
            ref, got = _both(monkeypatch, rows, g, n, f32, tile)
            assert torch.equal(ref, got), (m, n, k, tile)
