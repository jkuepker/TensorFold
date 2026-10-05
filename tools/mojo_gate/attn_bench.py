"""Phase 3 microbench: the Mojo tree attention (TF_ROCM_TREE_KERNEL=mojo) against the HIP WMMA kernels on the 27B's
attention geometry (24 query heads, 4 KV heads, head size 256), a drafted round's chain of verify rows over
``context`` committed keys, cold KV. Each kind runs a captured graph of attention calls over rotating cache copies
(512 MB together at least, past the R9700's 64 MB MALL, each read once a replay), timed with events; kinds alternate
over ROUNDS and the median counts. ``total`` is one attention call (shared + tail + merge); the shared, tail and merge
columns time each launch alone the same way. GB/s is the cache bytes (keys and values) a call reads over its time.

usage (container, card B, fleetq slot held): PYTHONPATH=src python tools/mojo_gate/attn_bench.py [--rows 12,1]
    [--context 16384,65536,131072] [--kv8]
"""

import argparse
import os
import statistics

import torch

from tensorfold.cuda.kernels import attention as shared

H, HK, D = 24, 4, 256
COLD_BYTES = 512 << 20
KINDS = ("wmma", "mojo")


def caches(p, copies, gen, kv8):
    from tensorfold.cuda.kernels import kv8 as packing

    out = []
    for _ in range(copies):
        kc = torch.randn((p + 128, HK, D), generator=gen, device="cuda").bfloat16()
        vc = torch.randn((p + 128, HK, D), generator=gen, device="cuda").bfloat16()
        if kv8:
            kc, vc = packing.reference_pack(kc), packing.reference_pack(vc)
        out.append((kc, vc))
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


def use(kind):
    os.environ["TF_ROCM_TREE_KERNEL"] = kind
    shared._rocm_kernel.cache_clear()
    return shared._mojo(torch.cuda.current_device()) if kind == "mojo" else shared._rocm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="12,1")
    ap.add_argument("--context", default="16384,65536,131072")
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--kv8", action="store_true")
    a = ap.parse_args()
    prop = torch.cuda.get_device_properties(0)
    print(f"# {prop.name} pci_bus_id={prop.pci_bus_id} {prop.gcnArchName} torch {torch.__version__} "
          f"kv={'fp8' if a.kv8 else 'bf16'}")
    assert prop.pci_bus_id == 7, "not card B"
    gen = torch.Generator(device="cuda").manual_seed(0)
    print("| context | rows | part | wmma us | mojo us | wmma GB/s | mojo GB/s | mojo/wmma |")
    print("|---|---|---|---|---|---|---|---|")
    worst = 0.0
    scale = 1 / 16
    tails = 1 + -(-shared.MAX_NODES // shared.CHUNK)
    for p in [int(x) for x in a.context.split(",")]:
        row_bytes = (D + 16) if a.kv8 else 2 * D       # a packed FP8 row (kv8.ROW8), or bf16
        nbytes = 2 * p * HK * row_bytes                 # keys and values a call reads
        copies = max(2, -(-COLD_BYTES // nbytes))
        cs = caches(p, copies, gen, a.kv8)
        offs = [torch.tensor(shared.offsets([c], "cuda"), dtype=torch.int64, device="cuda").view(-1, 2) for c in cs]
        for w in [int(r) for r in a.rows.split(",")]:
            q = torch.randn((w, H, D), generator=gen, device="cuda").bfloat16()
            kn = torch.randn((w, HK, D), generator=gen, device="cuda").bfloat16()
            vn = torch.randn((w, HK, D), generator=gen, device="cuda").bfloat16()
            plan = shared.plan([[-1] + list(range(w - 1))], [p], H // HK, "cuda")
            po = torch.empty((plan.chunks, w, H, D), device="cuda")
            pm = torch.empty((plan.chunks, w, H), device="cuda")
            pl = torch.empty_like(pm)
            out = torch.empty_like(q)
            launch = shared.rocm_launch(-(-w * (H // HK) // shared.QUERY_TILE), plan.chunks)
            calls = max(a.calls, copies)
            origin = shared.base("cuda")
            outs = {}
            graphs = {}
            for kind in KINDS:
                ext = use(kind)
                outs[kind] = shared.attention(q, kn, vn, offs[0], plan, scale=scale, kv8=a.kv8)

                def total(i, ext=ext):
                    shared.attention(q, kn, vn, offs[i % copies], plan, scale=scale, kv8=a.kv8)

                def part_shared(i, ext=ext):
                    ext.shared(q, origin, offs[i % copies], plan.streams, plan.items, po, pm, pl, HK, *launch, scale,
                               a.kv8)

                def part_tail(i, ext=ext):
                    ext.tail(q, kn, vn, origin, offs[i % copies], plan.streams, plan.rows, plan.paths, plan.depths,
                             po, pm, pl, tails, scale, a.kv8)

                def part_merge(i, kind=kind, ext=ext):
                    if kind == "mojo":
                        ext.merge(po, pm, pl, out, plan.streams, plan.rows)
                    else:
                        shared._merge[(w, HK, D // shared.MERGE_COLUMNS)](po, pm, pl, out, plan.streams, plan.rows,
                                                                          w, H=H, D=D, G=H // HK,
                                                                          DS=shared.MERGE_COLUMNS, num_warps=4)

                graphs[kind] = {name: graph(fn, calls) for name, fn in
                                (("total", total), ("shared", part_shared), ("tail", part_tail),
                                 ("merge", part_merge))}
            assert torch.equal(outs["wmma"].view(torch.int16), outs["mojo"].view(torch.int16)), "bits differ"
            for part in ("total", "shared", "tail", "merge"):
                ts = {k: [] for k in KINDS}
                for _ in range(a.rounds):
                    for k in KINDS:
                        ts[k].append(timed(graphs[k][part], calls))
                tw, tm = statistics.median(ts["wmma"]), statistics.median(ts["mojo"])
                if part == "total":
                    worst = max(worst, tm / tw)
                gb = (lambda t: f"{nbytes / t / 1e3:.0f}") if part in ("total", "shared") else (lambda t: "-")
                print(f"| {p} | {w} | {part} | {tw:.1f} | {tm:.1f} | {gb(tw)} | {gb(tm)} | {tm / tw:.3f} |",
                      flush=True)
            del graphs, po, pm, pl
            torch.cuda.empty_cache()
        del cs, offs
        torch.cuda.empty_cache()
    print(f"# worst total mojo/wmma {worst:.3f}")


if __name__ == "__main__":
    main()
