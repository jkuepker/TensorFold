"""The Mojo lane matmul (qmm_rocm.mojo, TF_ROCM_LANE=mojo) gives the HIP WMMA lane's bits, row for row."""

import shutil

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import gfx12, hip  # noqa: E402

if not hip() or not gfx12():
    pytest.skip("the WMMA lane matmul is gfx12's", allow_module_level=True)

from tensorfold.cuda import mojo  # noqa: E402

try:
    mojo.mojo_binary()
except RuntimeError:
    pytest.skip("no Mojo compiler", allow_module_level=True)

from tensorfold.cuda.kernels import qmm_groups  # noqa: E402

# test_qmm_rocm.py's shapes, then the 27B's projections as the decode path runs them (K = 5120 unless noted):
# attention q, k/v, [k | v], [q | k | v], o; GDN qkv, z, a/b, [z | b | a], [qkv | z | b | a], out; MLP gate/up,
# [gate | up], down
TEST_SHAPES = [(1000, 1024), (48, 5120), (5120, 17408), (17408, 5120), (130, 64), (1, 128)]
SHAPES_27B = [(12288, 5120), (1024, 5120), (2048, 5120), (14336, 5120), (5120, 6144), (10240, 5120), (6144, 5120),
              (48, 5120), (6240, 5120), (16480, 5120), (17408, 5120), (34816, 5120), (5120, 17408)]


def _weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    words = words.to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
    return qmm_groups.to_groups(words, scales, biases), gen


def _same(x, g, n, f32=False):
    hip_ = qmm_groups.gemv(x, *g, n, f32=f32, wmma=True)
    mojo_ = qmm_groups.gemv(x, *g, n, f32=f32, mojo=True)
    return torch.equal(hip_.view(torch.int16 if not f32 else torch.int32),
                       mojo_.view(torch.int16 if not f32 else torch.int32))


@pytest.mark.parametrize("n,k", TEST_SHAPES)
def test_mojo_gives_the_wmma_bits_on_the_rocm_test_shapes(n, k):
    g, gen = _weight(n, k, n + k)
    x = torch.randn(40, k, generator=gen, device="cuda").bfloat16()
    for m in (1, 4, 8, 12, 16, 17, 33, 40):
        assert _same(x[:m].contiguous(), g, n), m
        assert _same(x[:m].contiguous(), g, n, f32=True), m


@pytest.mark.parametrize("n,k", SHAPES_27B)
def test_mojo_gives_the_wmma_bits_on_the_27b_projections(n, k):
    g, gen = _weight(n, k, 7 * n + k)
    x = torch.randn(16, k, generator=gen, device="cuda").bfloat16()
    for m in (1, 4, 8, 12, 16):
        assert _same(x[:m].contiguous(), g, n), m
    assert _same(x[:12].contiguous(), g, n, f32=True)


def test_mojo_runs_on_the_current_stream():
    n, k = 5120, 17408
    g, gen = _weight(n, k, 3)
    x = torch.randn(12, k, generator=gen, device="cuda").bfloat16()
    ref = qmm_groups.gemv(x, *g, n, wmma=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        big = torch.randn(4096, 4096, device="cuda")
        for _ in range(4):
            big = big @ big                             # keeps the stream busy before x changes on it
        y = x * 1.0                                     # written on the side stream just before the matmul
        got = qmm_groups.gemv(y, *g, n, mojo=True)
    side.synchronize()
    assert torch.equal(got, ref)


def test_mojo_lane_is_picked_by_tf_rocm_lane(monkeypatch):
    monkeypatch.setenv("TF_ROCM_LANE", "mojo")
    qmm_groups.lane_kernel.cache_clear()
    try:
        assert qmm_groups.lane_kernel() == "mojo"
        g, gen = _weight(1024, 5120, 9)
        x = torch.randn(12, 5120, generator=gen, device="cuda").bfloat16()
        assert torch.equal(qmm_groups.matmul(x, *g, 1024), qmm_groups.gemv(x, *g, 1024, wmma=True))
    finally:
        qmm_groups.lane_kernel.cache_clear()


@pytest.mark.parametrize("slices,total", [(2, 1), (3, 1000), (16, 70001)])
@pytest.mark.parametrize("f32", [False, True])
def test_mojo_reduce_adds_slices_in_order(slices, total, f32):
    part = torch.randn(slices, total, device="cuda") * 1e3
    out = torch.empty(total, device="cuda", dtype=torch.float32 if f32 else torch.bfloat16)
    qmm_groups._mojo(out.get_device()).reduce_slices(part, slices, out)
    ref = part[0].clone()
    for s in range(1, slices):
        ref = ref + part[s]
    assert torch.equal(out, ref if f32 else ref.bfloat16())


def test_missing_mojo_says_so(monkeypatch):
    monkeypatch.delenv("TF_MOJO", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(mojo.sys, "executable", "/nonexistent/python")
    mojo.mojo_binary.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="needs the Mojo compiler"):
            mojo.mojo_binary()
    finally:
        mojo.mojo_binary.cache_clear()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_mojo_gives_the_wmma_bits_on_each_gpu():
    """One process, two GPUs: each loads its own module, and a launch from the other GPU's context stays on its
    tensors' GPU."""

    n, k = 1000, 1024
    for dev in (0, 1, 0):
        with torch.cuda.device(dev):
            g, gen = _weight(n, k, n + k)
            x = torch.randn(40, k, generator=gen, device="cuda").bfloat16()
            for m in (1, 12, 33, 40):
                assert _same(x[:m].contiguous(), g, n), (dev, m)
                assert _same(x[:m].contiguous(), g, n, f32=True), (dev, m)
    with torch.cuda.device(1):
        g, gen = _weight(n, k, n + k)
        x = torch.randn(12, k, generator=gen, device="cuda").bfloat16()
        xs = qmm_groups.group_sums(x)
        want = qmm_groups.gemv(x, *g, n, xs, wmma=True)
    with torch.cuda.device(0):
        got = qmm_groups.gemv(x, *g, n, xs, mojo=True)
    assert torch.equal(want.view(torch.int16), got.view(torch.int16))
