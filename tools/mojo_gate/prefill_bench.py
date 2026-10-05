"""Mojo vs reference prompt kernels on the 27B's shapes (R9700, card B).

gemm: the bf16 prompt GEMM after the weight's rounding (Triton ``_gemm`` vs ``prefill_rocm.mojo``), each the GEMM
kernel alone on the same bf16 weight, rows of a prompt chunk; TFLOPS = 2 M N K / time. Median of ROUNDS timed
launches (CUDA events around REPS back-to-back launches), the kernels' rounds interleaved, weights warm (compute
bound: 2341 rows reuse each weight 19 times). "x" is the reference's time over Mojo's (above 1: Mojo faster).

attn: prompt attention (attention_rocm.cu's prompt_kernel vs attention_rocm.mojo's prompt kernels) at the 27B's
geometry (24 query heads, 4 KV heads, head size 256) on the prompt chunks the engine takes on the R9700 (2560-row
chunks: the last chunk of a 4k, 16k and 30k prompt, plus the first), bf16 or packed FP8 caches, RB and loaders as
rocm_rows picks them. Effective TFLOPS = 4 H D (W p0 + W (W + 1) / 2) / time: QK^T and PV over the keys each row
attends (causal), the masked part of the diagonal tiles not counted.

gemm8: the FP8 prompt GEMM (Triton ``_gemm8`` vs the Mojo gemm8 kernels) on e4m3 rows and the weight's e4m3 codes,
the codes' widening (``_nibbles8``) left out of both; TFLOPS as for gemm.

usage (container, slot): PYTHONPATH=src python3 tools/mojo_gate/prefill_bench.py gemm|gemm8 [--rows 2341] [--tiles ..]
                         PYTHONPATH=src python3 tools/mojo_gate/prefill_bench.py attn
"""

from __future__ import annotations

import argparse
import statistics

import torch
import triton

from tensorfold.cuda.kernels import qmm_groups

SHAPES = [("attn q", 12288, 5120), ("attn kv", 2048, 5120), ("attn o / gdn out", 5120, 6144),
          ("gdn qkv", 10240, 5120), ("gdn zba", 6240, 5120), ("mlp gate / up", 17408, 5120),
          ("mlp down", 5120, 17408)]
ROUNDS = 7
REPS = 5


def timed(fns: list) -> list[float]:
    """Median seconds a call of each of ``fns``, their rounds interleaved (clocks drift between runs: compare kernels
    inside one interleaved measurement, never across)."""

    for fn in fns:
        for _ in range(3):
            fn()
    times: list[list[float]] = [[] for _ in fns]
    for _ in range(ROUNDS):
        for i, fn in enumerate(fns):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(REPS):
                fn()
            b.record()
            b.synchronize()
            times[i].append(a.elapsed_time(b) / 1e3 / REPS)
    return [statistics.median(t) for t in times]


def triton_gemm(x, w, out):
    m, k = x.shape
    n = w.shape[0]
    bm, bn, bk, warps, stages = qmm_groups.prefill_config()
    qmm_groups._gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](x, w, out, m, N=n, K=k, BM=bm, BN=bn, BK=bk,
                                                               num_warps=warps, num_stages=stages)


def bench_gemm(rows: list[int], tiles: list[str]) -> None:
    ext = qmm_groups._prefill_mojo(torch.cuda.current_device())
    names = list(qmm_groups.PREFILL_TILES)
    gen = torch.Generator(device="cuda").manual_seed(1)
    head = f"{'shape':<18} {'N':>6} {'K':>6} {'M':>5} {'triton':>14}" + "".join(f" {'mojo ' + t:>22}" for t in tiles)
    print(head)
    for m in rows:
        for name, n, k in SHAPES:
            w = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).bfloat16()
            x = torch.randn(m, k, generator=gen, device="cuda").bfloat16()
            ref = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            flops = 2 * m * n * k
            outs = [torch.empty_like(ref) for _ in tiles]
            fns = [lambda: triton_gemm(x, w, ref)] + [
                (lambda o, which: (lambda: ext.gemm(x, w, o, which)))(o, names.index(t)) for o, t in zip(outs, tiles)]
            t_ref, *t_mojo = timed(fns)
            line = f"{name:<18} {n:6d} {k:6d} {m:5d} {t_ref * 1e6:7.0f}us {flops / t_ref / 1e12:5.1f}T"
            for o, t_m in zip(outs, t_mojo):
                same = torch.equal(ref, o)
                line += f" {t_m * 1e6:7.0f}us {flops / t_m / 1e12:5.1f}T x{t_ref / t_m:4.2f}{'' if same else ' BITS!'}"
            print(line, flush=True)


def bench_gemm8(rows: list[int], tiles: list[str]) -> None:
    ext = qmm_groups._prefill_mojo(torch.cuda.current_device())
    names = list(qmm_groups.PREFILL8_TILES)           # launcher index: after the bf16 tiles
    gen = torch.Generator(device="cuda").manual_seed(1)
    print(f"{'shape':<18} {'N':>6} {'K':>6} {'M':>5} {'triton':>14}" + "".join(f" {'mojo ' + t:>22}" for t in tiles))
    for m in rows:
        for name, n, k in SHAPES:
            words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda",
                                  dtype=torch.int64).to(torch.int32)
            scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
            biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
            words, scales, biases = qmm_groups.to_groups(words, scales, biases)
            w8 = torch.empty((n, k), dtype=torch.uint8, device="cuda")
            qmm_groups._nibbles8[(k // 64, triton.cdiv(n, 64))](words, w8, N=n, K=k, BLOCK_N=64, num_warps=4)
            x = torch.randn(m, k, generator=gen, device="cuda")
            a = x.abs().amax(1) / 448.0
            x8 = (x / a[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
            xs = (x.view(m, k // 64, 64).sum(2) / a[:, None]).bfloat16()
            ref = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            bm, bn, warps, stages = qmm_groups.prefill8_config()

            def tri():
                qmm_groups._gemm8[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                    x8, xs, a, w8, scales, biases, ref, m, N=n, K=k, BM=bm, BN=bn, KB=16,
                    GROUP=qmm_groups.prefill8_group(), num_warps=warps, num_stages=stages)

            outs = [torch.empty_like(ref) for _ in tiles]
            fns = [tri] + [(lambda o, which: (lambda: ext.gemm8(x8, xs, a, w8, scales, biases, o, which)))(
                o, len(qmm_groups.PREFILL_TILES) + names.index(t)) for o, t in zip(outs, tiles)]
            t_ref, *t_mojo = timed(fns)
            flops = 2 * m * n * k
            line = f"{name:<18} {n:6d} {k:6d} {m:5d} {t_ref * 1e6:7.0f}us {flops / t_ref / 1e12:5.1f}T"
            for o, t_m in zip(outs, t_mojo):
                same = torch.equal(ref, o)
                line += f" {t_m * 1e6:7.0f}us {flops / t_m / 1e12:5.1f}T x{t_ref / t_m:4.2f}{'' if same else ' BITS!'}"
            print(line, flush=True)


ATTN_CHUNKS = [("4k first", 0, 2048), ("4k last", 2048, 2048), ("16k first", 0, 2340), ("16k last", 14043, 2341),
               ("30k first", 0, 2500), ("30k last", 27500, 2500)]


def bench_attn() -> None:
    from tensorfold.cuda.kernels import attention as tree
    from tensorfold.cuda.kernels import kv8, prefill_attention

    h, hk, d = 24, 4, 256
    hip_ext, mojo_ext = prefill_attention._rocm(), tree._mojo(torch.cuda.current_device())
    gen = torch.Generator(device="cuda").manual_seed(2)
    print(f"{'chunk':<10} {'p0':>6} {'W':>5} {'cache':>5} {'hip':>18} {'mojo':>24}")
    for packed in (False, True):
        for name, p0, w in ATTN_CHUNKS:
            k = (torch.randn(p0 + w, hk, d, generator=gen, device="cuda") * 1.5).bfloat16()
            v = torch.randn(p0 + w, hk, d, generator=gen, device="cuda").bfloat16()
            if packed:
                k, v = kv8.pack(k)[1], kv8.pack(v)[1]
            q = (torch.randn(w, h, d, generator=gen, device="cuda") * 1.5).bfloat16()
            ref, got = torch.empty_like(q), torch.empty_like(q)
            rb, pipe = prefill_attention.rocm_rows(w)
            sc = d ** -0.5
            t_ref, t_m = timed([lambda: hip_ext.attention(q, k, v, ref, p0, sc, rb, pipe),
                                lambda: mojo_ext.attention(q, k, v, got, p0, sc, rb, pipe)])
            flops = 4 * h * d * (w * p0 + w * (w + 1) / 2)
            same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
            cache = "fp8" if packed else "bf16"
            print(f"{name:<10} {p0:6d} {w:5d} {cache:>5} {t_ref * 1e6:9.0f}us {flops / t_ref / 1e12:5.1f}T"
                  f" {t_m * 1e6:9.0f}us {flops / t_m / 1e12:5.1f}T x{t_ref / t_m:4.2f}{'' if same else ' BITS!'}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gemm", "gemm8", "attn"])
    ap.add_argument("--rows", default="2341")
    ap.add_argument("--tiles", default="")
    args = ap.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} pci_bus_id={props.pci_bus_id} {props.gcnArchName}")
    if args.what == "gemm":
        bench_gemm([int(r) for r in args.rows.split(",")],
                   (args.tiles or ",".join(qmm_groups.PREFILL_TILES)).split(","))
    elif args.what == "gemm8":
        bench_gemm8([int(r) for r in args.rows.split(",")],
                    (args.tiles or ",".join(qmm_groups.PREFILL8_TILES)).split(","))
    elif args.what == "attn":
        bench_attn()


if __name__ == "__main__":
    main()
