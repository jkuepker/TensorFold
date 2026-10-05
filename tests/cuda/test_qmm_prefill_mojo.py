"""The Mojo prompt matmuls (qmm_prefill.mojo, TF_CUDA_PREFILL_GEMM=mojo) give the CUDA kernels' bits, row for row:
bf16 rows (qmm_prefill.cu) and FP8 rows (--prefill-fp8, qmm_prefill8.cu)."""

import pytest
import torch

if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
    pytest.skip("NVIDIA only", allow_module_level=True)

from tensorfold.cuda import mojo  # noqa: E402

try:
    mojo.mojo_binary()
except RuntimeError:
    pytest.skip("no Mojo compiler", allow_module_level=True)

from tensorfold.cuda.kernels import qmm as shared  # noqa: E402

# test_qwen27_prefill.py's and test_qmm.py's prompt shapes, then the 27B's projections as the prompt path runs them
# (K = 5120 unless noted): attention q (with its gate), k/v, [k | v], o; GDN qkv, z, a/b, [z | b | a], out; MLP
# gate/up, down; the LM head's rows
TEST_SHAPES = [(1000, 1024), (1100, 1024), (128, 5120), (130, 64), (1, 128)]
SHAPES_27B = [(12288, 5120), (1024, 5120), (2048, 5120), (5120, 6144), (10240, 5120), (6144, 5120), (48, 5120),
              (6240, 5120), (17408, 5120), (5120, 17408)]
# chunks() splits a prompt into even chunks of at most 4096 rows, so any count up to 4096 occurs
ROWS = (1, 7, 16, 127, 128, 129, 333, 1000, 2731, 4096)


def _weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
    return shared.pack(words.to(torch.int32), scales, biases, 64), gen


def _same(x, q, f32=False, fp8=False):
    if fp8:
        xq = shared.quantize_rows(x)
        ref = shared.prefill_matmul8(xq, q, f32=f32, mojo=False)
        got = shared.prefill_matmul8(xq, q, f32=f32, mojo=True)
    else:
        ref = shared.prefill_matmul(x, q, f32=f32, mojo=False)
        got = shared.prefill_matmul(x, q, f32=f32, mojo=True)
    view = torch.int32 if f32 else torch.int16
    return torch.equal(ref.view(view), got.view(view))


@pytest.fixture(params=[False, True], ids=["bf16", "fp8"])
def fp8(request):
    return request.param


def test_one_group_one_tile(fp8):
    q, gen = _weight(128, 64, 1)
    x = torch.randn(16, 64, generator=gen, device="cuda").bfloat16()
    assert _same(x, q, fp8=fp8) and _same(x, q, f32=True, fp8=fp8)


@pytest.mark.parametrize("n,k", TEST_SHAPES)
def test_mojo_gives_the_cuda_bits_on_the_test_shapes(n, k, fp8):
    q, gen = _weight(n, k, n + k)
    x = torch.randn(1100, k, generator=gen, device="cuda").bfloat16()
    for m in (1, 7, 16, 64, 129, 333, 1100):
        assert _same(x[:m].contiguous(), q, fp8=fp8), m
        assert _same(x[:m].contiguous(), q, f32=True, fp8=fp8), m


@pytest.mark.parametrize("n,k", SHAPES_27B)
def test_mojo_gives_the_cuda_bits_on_the_27b_projections(n, k, fp8):
    q, gen = _weight(n, k, 7 * n + k)
    x = torch.randn(max(ROWS), k, generator=gen, device="cuda").bfloat16()
    for m in ROWS:
        assert _same(x[:m].contiguous(), q, fp8=fp8), m
    assert _same(x[:333].contiguous(), q, f32=True, fp8=fp8)


def test_strided_rows():
    """Rows of a wider tensor (the engine's column slices): the row stride is read, not assumed."""
    q, gen = _weight(1024, 5120, 5)
    wide = torch.randn(300, 5120 + 64, generator=gen, device="cuda").bfloat16()
    x = wide[:, 64:]
    assert _same(x, q)


def test_zero_rows():
    q, _ = _weight(1024, 5120, 4)
    x = torch.empty(0, 5120, device="cuda", dtype=torch.bfloat16)
    assert shared.prefill_matmul(x, q, mojo=True).shape == (0, 1024)


def test_mojo_runs_on_the_current_stream():
    q, gen = _weight(5120, 17408, 3)
    x = torch.randn(512, 17408, generator=gen, device="cuda").bfloat16()
    ref = shared.prefill_matmul(x, q, mojo=False)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        big = torch.randn(4096, 4096, device="cuda")
        for _ in range(4):
            big = big @ big                             # keeps the stream busy before x changes on it
        y = x * 1.0                                     # written on the side stream just before the matmul
        got = shared.prefill_matmul(y, q, mojo=True)
    side.synchronize()
    assert torch.equal(got, ref)


def test_mojo_is_picked_by_tf_cuda_prefill_gemm(monkeypatch):
    q, gen = _weight(1024, 5120, 9)
    x = torch.randn(200, 5120, generator=gen, device="cuda").bfloat16()
    ref = shared.prefill_matmul(x, q, mojo=False)
    monkeypatch.setenv("TF_CUDA_PREFILL_GEMM", "mojo")
    shared.prefill_kernel.cache_clear()
    try:
        assert shared.prefill_kernel() == "mojo"
        assert torch.equal(shared.prefill_matmul(x, q), ref)
        xq = shared.quantize_rows(x)
        assert torch.equal(shared.prefill_matmul8(xq, q), shared.prefill_matmul8(xq, q, mojo=False))
        monkeypatch.setenv("TF_CUDA_PREFILL_GEMM", "triton")
        shared.prefill_kernel.cache_clear()
        with pytest.raises(ValueError, match="TF_CUDA_PREFILL_GEMM"):
            shared.prefill_kernel()
    finally:
        shared.prefill_kernel.cache_clear()
