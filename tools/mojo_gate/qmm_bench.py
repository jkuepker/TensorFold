"""Phase 1 microbench: the Mojo lane matmul (TF_ROCM_LANE=mojo) against the HIP WMMA lane on the 27B's decode
projections, cold weights. Each lane runs a captured graph of CALLS matmuls over rotating weight copies (together
past the R9700's 64 MB MALL, so every call streams its weights from memory), timed with events; lanes alternate over
ROUNDS and the median counts. GB/s is the weight bytes (words + scales + biases) a call reads over its time.

usage (container, card B, fleetq slot held): PYTHONPATH=src python tools/mojo_gate/qmm_bench.py [--rows 1,4,12,16]
"""

import argparse
import statistics

import torch

from tensorfold.cuda.kernels import qmm_groups

SHAPES = [  # (name, N, K) as the decode path runs them
    ("attn q", 12288, 5120), ("attn k/v", 1024, 5120), ("attn kv", 2048, 5120), ("attn qkv", 14336, 5120),
    ("attn o / gdn out", 5120, 6144), ("gdn qkv", 10240, 5120), ("gdn z", 6144, 5120), ("gdn a/b", 48, 5120),
    ("gdn zba", 6240, 5120), ("gdn proj", 16480, 5120), ("mlp gate/up", 17408, 5120), ("mlp gu", 34816, 5120),
    ("mlp down", 5120, 17408),
]
COLD_BYTES = 512 << 20


def weights(n, k, copies, gen):
    out = []
    for _ in range(copies):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda", dtype=torch.int64)
        scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.01 + 0.001).bfloat16()
        biases = (torch.randn(n, k // 64, generator=gen, device="cuda") * 0.02).bfloat16()
        out.append(qmm_groups.to_groups(words.to(torch.int32), scales, biases))
    return out


def graph(fn, calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)                                       # warm: kernels loaded, allocator primed
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i)
    return g


def timed(g, calls):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / calls            # us a call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,4,12,16")
    ap.add_argument("--calls", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"# {p.name} pci_bus_id={p.pci_bus_id} {p.gcnArchName} torch {torch.__version__}")
    assert p.pci_bus_id == 7, "not card B"
    gen = torch.Generator(device="cuda").manual_seed(0)
    rows = [int(r) for r in a.rows.split(",")]
    print("| shape | N | K | rows | wmma us | mojo us | wmma GB/s | mojo GB/s | mojo/wmma |")
    print("|---|---|---|---|---|---|---|---|---|")
    worst = 0.0
    for name, n, k in SHAPES:
        if a.only and a.only not in name:
            continue
        nbytes = n * k // 2 + 2 * 2 * n * (k // 64)
        copies = max(2, -(-COLD_BYTES // nbytes))
        ws = weights(n, k, copies, gen)
        for m in rows:
            x = torch.randn(m, k, generator=gen, device="cuda").bfloat16()
            xs = qmm_groups.group_sums(x)
            assert torch.equal(qmm_groups.gemv(x, *ws[0], n, xs, wmma=True), qmm_groups.gemv(x, *ws[0], n, xs, mojo=True))
            gw = graph(lambda i: qmm_groups.gemv(x, *ws[i % copies], n, xs, wmma=True), a.calls)
            gm = graph(lambda i: qmm_groups.gemv(x, *ws[i % copies], n, xs, mojo=True), a.calls)
            tw, tm = [], []
            for _ in range(a.rounds):
                tw.append(timed(gw, a.calls))
                tm.append(timed(gm, a.calls))
            w, mj = statistics.median(tw), statistics.median(tm)
            worst = max(worst, mj / w)
            print(f"| {name} | {n} | {k} | {m} | {w:.1f} | {mj:.1f} | {nbytes / w / 1e3:.0f} | {nbytes / mj / 1e3:.0f} "
                  f"| {mj / w:.3f} |", flush=True)
            del gw, gm
        del ws
        torch.cuda.empty_cache()
    print(f"# worst mojo/wmma {worst:.3f}")


if __name__ == "__main__":
    main()
