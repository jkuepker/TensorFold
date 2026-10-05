"""Per-kernel GPU time of one Qwen3.8-27B prompt prefill on the R9700, grouped by role.

The model loads as the gfx12 engine loads it (``load`` + ``prepare(fuse=True)``, the card's prompt chunk rows), one
warm prefill runs first (Triton, HIP and Mojo builds), one timed prefill gives the wall tok/s and a third runs under
torch.profiler. Every prompt matmul, attention call, DeltaNet chain and glue call is wrapped in a record_function, so
each kernel is charged to the call that launched it (GEMMs by their (N, K)).

usage (container, card B): python3 tools/mojo_gate/prefill_profile.py --tokens 4096,16384 [--fp8]
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

MODEL = os.environ.get("TENSORFOLD_MLX_MODEL", "/models/Qwen3.8-27B-MLX-4bit")


_depth = [0]


def _wrap(module, name, label):
    """Calls of ``module.name`` in a named range; a call inside another wrapped call is charged to the outer one."""

    fn = getattr(module, name)

    @functools.wraps(fn)
    def inner(*a, **kw):
        if _depth[0]:
            return fn(*a, **kw)
        _depth[0] += 1
        try:
            with record_function(label(*a, **kw) if callable(label) else label):
                return fn(*a, **kw)
        finally:
            _depth[0] -= 1

    setattr(module, name, inner)


def _rows(x):
    return (x[0] if isinstance(x, tuple) else x).shape


def _mm_label(x, w, f32=False):
    m, k = _rows(x)
    n = w.n if getattr(w, "n", None) else w.rows
    return f"gemm N={n} K={k}"


def instrument():
    from tensorfold.cuda.kernels import gdn
    from tensorfold.families.qwen3_5.cuda import glue, prefill, prefill_bf16, prefill_glue

    _wrap(prefill, "_mm", _mm_label)
    _wrap(prefill, "attention", "prompt attention")
    _wrap(gdn, "chain", "gdn chain")
    for mod, tag in ((glue, "glue"), (prefill_bf16, "glue"), (prefill_glue, "glue")):
        for name in dir(mod):
            fn = getattr(mod, name)
            if name.startswith("_") or not callable(fn) or getattr(fn, "__module__", None) != mod.__name__:
                continue
            if isinstance(fn, type):
                continue
            _wrap(mod, name, f"{tag} {name}")


def load():
    from tensorfold.cuda.capacity import config, total_bytes
    from tensorfold.cuda.geometry import prompt_row_bytes, prompt_rows
    from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare
    from tensorfold.families.qwen3_5.cuda.weights import load as load_weights

    torch.cuda.set_device(0)
    chunk = prompt_rows(total_bytes(torch), prompt_row_bytes(config(MODEL), 1))
    w = load_weights(MODEL)
    prepare(w, fuse=True)
    w.prompt_rows = chunk
    return w, chunk


def category(key: str) -> str:
    if key.startswith("gemm"):
        return key
    if key == "prompt attention":
        return "prompt attention"
    if key == "gdn chain":
        return "gdn chain"
    if key in ("glue gdn_pre",):
        return "gdn pre (conv, gates)"
    return "norms/glue: " + key.split(" ", 1)[1]


def run(w, n: int, seed: int = 3):
    from tensorfold.families.qwen3_5.cuda.forward import State
    from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, w.config.vocab, (n,), generator=g).tolist()
    st = State(w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prefill_state(w, prompt, st)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del st
    return dt


LABELS = ("gemm", "prompt attention", "gdn chain", "glue ")


def profile_one(w, n: int):
    run(w, n)                                    # warm: builds, autotune caches, allocator
    wall = min(run(w, n) for _ in range(2))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run(w, n)
    events = prof.key_averages()
    kernels = collections.Counter()
    calls = collections.Counter()
    ranges = collections.Counter()
    names = {e.key for e in events if e.key.startswith(LABELS)}
    for e in events:
        dev = getattr(e, "self_device_time_total", 0) or 0
        # the ranges' own GPU-side annotation events carry their kernels' time again: skip them
        if e.device_type == torch.autograd.DeviceType.CUDA and dev and e.key not in names:
            kernels[e.key] += dev
            calls[e.key] += e.count
    for e in events:
        if e.key.startswith(LABELS):            # a range may show up as a CPU and a GPU entry: one of them
            ranges[e.key] = max(ranges[e.key], getattr(e, "device_time_total", 0) or 0)
    total = sum(kernels.values())
    return wall, total, kernels, calls, ranges


def report(tag: str, n: int, chunk: int, wall, total, kernels, calls, ranges) -> str:
    lines = [f"== {tag}: {n} tokens, chunk rows {chunk}; wall {wall:.3f} s = {n / wall:,.0f} tok/s; "
             f"GPU kernel time {total / 1e3:,.1f} ms =="]
    cats = collections.Counter()
    for k, v in ranges.items():
        cats[category(k)] += v
    lines.append("-- by role (device time of kernels launched inside each call) --")
    gemm = sum(v for k, v in cats.items() if k.startswith("gemm"))
    groups = collections.Counter({"projection GEMMs (all)": gemm})
    for k, v in cats.items():
        if not k.startswith("gemm"):
            groups[k.split(":")[0]] += v
    groups["other (not in a wrapped call)"] = total - sum(cats.values())
    for k, v in groups.most_common():
        lines.append(f"  {k:<40} {v / 1e3:10.1f} ms  {100 * v / total:5.1f}%")
    lines.append("-- GEMMs by shape --")
    for k, v in sorted(((k, v) for k, v in cats.items() if k.startswith("gemm")), key=lambda t: -t[1]):
        lines.append(f"  {k:<40} {v / 1e3:10.1f} ms  {100 * v / total:5.1f}%")
    lines.append("-- norms/glue by call --")
    for k, v in sorted(((k, v) for k, v in cats.items() if k.startswith("norms")), key=lambda t: -t[1]):
        lines.append(f"  {k:<40} {v / 1e3:10.1f} ms  {100 * v / total:5.1f}%")
    lines.append("-- kernels (self device time) --")
    for k, v in kernels.most_common(25):
        lines.append(f"  {k[:70]:<70} {calls[k]:6d}x {v / 1e3:10.1f} ms  {100 * v / total:5.1f}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,16384")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    from tensorfold.cuda import prompt_precision

    prompt_precision.set_fp8(args.fp8)          # before any weight loads, as the CLI sets it
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} pci_bus_id={props.pci_bus_id} {props.gcnArchName}", flush=True)
    w, chunk = load()
    print(f"fast_prefill={w.fast_prefill} fp8={prompt_precision.fp8()} chunk={chunk}", flush=True)
    instrument()
    tag = "FP8 prompts (--prefill-fp8)" if args.fp8 else "bf16 prompts (default)"
    env = {k: v for k, v in os.environ.items() if k.startswith("TF_ROCM")}
    out = {}
    for n in (int(t) for t in args.tokens.split(",")):
        wall, total, kernels, calls, ranges = profile_one(w, n)
        print(report(f"{tag} {env or ''}", n, chunk, wall, total, kernels, calls, ranges), flush=True)
        out[n] = dict(wall=wall, total=total, kernels=dict(kernels), ranges=dict(ranges))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
