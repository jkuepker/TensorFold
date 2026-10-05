"""The 27B's prompt matmul on the GB10: Mojo (qmm_prefill.mojo) against the CUDA reference (qmm_prefill.cu, tile 9),
TFLOPS per projection at 512-4096 prompt rows; speedup = cuda time / mojo time. Each timing is the median of TRIALS
trials of REPS back-to-back calls (CUDA events), the two kernels alternating trial by trial; outputs checked equal first.
``--tile T`` times the CUDA kernel at tile T instead of the engine's (prompt_tile).
usage (in the container, PYTHONPATH=src): python tools/mojo_gate/spark/prefill_bench.py [--reps N] [--trials N]
"""
import argparse
import statistics

import torch

from tensorfold.cuda.kernels import qmm as shared

# (name, n, k): the 27B's prompt projections (attention q with its gate, [k | v], o; GDN qkv, [z | b | a], out; MLP)
SHAPES = [("attn q", 12288, 5120), ("attn kv", 2048, 5120), ("attn o", 5120, 6144), ("gdn qkv", 10240, 5120),
          ("gdn zba", 6240, 5120), ("gdn out", 5120, 6144), ("mlp gate/up", 17408, 5120), ("mlp down", 5120, 17408)]
ROWS = (512, 1024, 2048, 4096)


def weight(n, k):
    gen = torch.Generator(device="cuda").manual_seed(n + k)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
    biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
    return shared.pack(words.to(torch.int32), scales, biases, 64)


def trial(call, reps):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        call()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--tile", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="three cases only (tuning)")
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"device {p.name} sm_{p.major}{p.minor} {p.multi_processor_count} SMs, torch {torch.__version__}")
    print(f"{'projection':12s} {'N':>6s} {'K':>6s} {'rows':>5s} {'cuda ms':>8s} {'mojo ms':>8s} {'cuda TF':>8s} "
          f"{'mojo TF':>8s} {'speedup':>9s}")
    ratios = []
    cases = [(s, m) for s in SHAPES for m in ROWS]
    if args.quick:
        cases = [(SHAPES[0], 4096), (SHAPES[1], 512), (SHAPES[7], 2048)]
    for (name, n, k), m in cases:
        q = weight(n, k)
        if True:
            x = torch.randn(m, k, device="cuda").bfloat16()
            out_c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            out_m = torch.empty_like(out_c)
            tile = shared.prompt_tile(m, q.n) if args.tile < 0 else args.tile   # the engine's _mm: 9 everywhere
            cuda = lambda: shared.prefill_matmul(x, q, out=out_c, tile=tile, mojo=False)  # noqa: E731
            mojo = lambda: shared.prefill_matmul(x, q, out=out_m, mojo=True)  # noqa: E731
            cuda()
            mojo()
            torch.cuda.synchronize()
            assert torch.equal(out_c, out_m), (name, m)
            for call in (cuda, mojo):
                trial(call, 3)
            tc, tm = [], []
            for _ in range(args.trials):
                tc.append(trial(cuda, args.reps))
                tm.append(trial(mojo, args.reps))
            c, mo = statistics.median(tc), statistics.median(tm)
            flop = 2.0 * m * n * k
            ratios.append(c / mo)
            print(f"{name:12s} {n:6d} {k:6d} {m:5d} {c:8.3f} {mo:8.3f} {flop / c / 1e9:8.1f} {flop / mo / 1e9:8.1f} "
                  f"{c / mo:9.3f}")
    print(f"speed mojo/cuda (cuda time / mojo time): min {min(ratios):.3f}, median {statistics.median(ratios):.3f}, "
          f"max {max(ratios):.3f}")


if __name__ == "__main__":
    main()
