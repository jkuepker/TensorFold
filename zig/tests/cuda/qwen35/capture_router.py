#!/usr/bin/env python3
"""The Qwen3.6 MoE router as the Zig engine's oracle: Triton launches of router, top-k and combine, with their inputs and outputs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

D, E, TOP_K = 2048, 256, 8
ROWS = (1, 2, 3, 5, 8, 15, 16, 17, 20, 31, 32, 33, 40, 48, 63, 64, 65, 100, 127, 128, 129, 200, 256, 2048)
COMBINE_ROWS = (1, 2, 3, 5, 8, 16, 17, 32, 33, 64, 100, 128, 129, 256)


def specialization() -> dict:
    """Each launched JIT function's parameter names and which ones Triton never specializes."""

    from triton.runtime.jit import JITFunction

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            out[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                "params": [p.name for p in fn.params],
                "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


class Dump:
    """Raw little-endian tensors by case and name, and an index of their shapes."""

    def __init__(self, root: Path) -> None:
        self.root, self.index = root, {}

    def save(self, case: str, name: str, t) -> None:
        import torch

        d = self.root / case
        d.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize()
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tofile(d / f"{name}.bin")
        self.index.setdefault(case, {})[name] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}


def tie_logits(gen, rows: int, device):
    """fp32 logits [rows, E + 1] on a coarse grid (many equal values), signed zeros and extreme shared-gate logits."""

    import torch

    grid = torch.randint(-6, 7, (rows, E + 1), generator=gen, device=device).float() * 0.25
    grid[:, 3] = grid[:, 200]                                   # an exact pair across the row
    grid[0, :E] = 0.0                                           # a row of equal logits
    if rows > 1:
        grid[1, :E] = -0.0
        grid[1, ::2] = 0.0                                      # mixed +0 and -0
    if rows > 2:
        grid[2, :E] = torch.randn(E, generator=gen, device=device)
        grid[2, 17] = grid[2, :E].max() + 1.0                   # a unique maximum
    extremes = torch.tensor([-100.0, -88.0, -20.0, -0.0, 0.0, 1e-3, 5.0, 20.0, 88.0, 100.0, 1e4, -1e4],
                            device=device)
    grid[:, E] = extremes[torch.arange(rows, device=device) % extremes.numel()]
    return grid.contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="folder of triton_aot_manifest.py")
    ap.add_argument("--seed", type=int, default=20261013)
    a = ap.parse_args()
    sys.path.insert(0, a.tools)
    import triton_aot_manifest as aot

    rec = aot.Recorder().install()
    import torch
    import triton

    from tensorfold.cuda import moe

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    dump = Dump(out / "fixtures")
    # router rows as the host dequant leaves them: bf16, the shared expert's gate row last
    w = (torch.randn((E + 1, D), generator=gen, device=dev) * 0.03).to(torch.bfloat16)
    w[5] = w[9]                                                 # two experts with equal logits for every input
    dump.save("weights", "rows", w)
    for m in ROWS:
        x = (torch.randn((m, D), generator=gen, device=dev) * (1.0 + 4.0 * (torch.rand((m, 1), generator=gen, device=dev) > 0.9))).to(torch.bfloat16)
        if m > 1:
            x[1] = 0.0                                          # every logit 0
        buf = SimpleNamespace(pick=torch.empty((m, TOP_K + 1), dtype=torch.int32, device=dev),
                              wts=torch.empty((m, TOP_K + 1), dtype=torch.float32, device=dev))
        logits = torch.empty((m, E + 1), dtype=torch.float32, device=dev)
        with rec.scope(f"router-{m}", detail=True):
            moe.router(x, w, logits)
            moe.select_rows(logits, buf, TOP_K, E)
        for name, t in (("x", x), ("logits", logits), ("pick", buf.pick), ("wts", buf.wts)):
            dump.save(f"router-{m}", name, t)
    for m in ROWS:
        logits = tie_logits(gen, m, dev)
        buf = SimpleNamespace(pick=torch.empty((m, TOP_K + 1), dtype=torch.int32, device=dev),
                              wts=torch.empty((m, TOP_K + 1), dtype=torch.float32, device=dev))
        with rec.scope(f"topk-{m}", detail=True):
            moe.select_rows(logits, buf, TOP_K, E)
        for name, t in (("logits", logits), ("pick", buf.pick), ("wts", buf.wts)):
            dump.save(f"topk-{m}", name, t)
    for m in COMBINE_ROWS:
        for dtype in (torch.float32, torch.bfloat16):
            y = (torch.randn((m, TOP_K + 1, D), generator=gen, device=dev) * 3.0).to(dtype)
            wts = torch.softmax(torch.randn((m, TOP_K + 1), generator=gen, device=dev), dim=1).to(torch.bfloat16).float().contiguous()
            tag = "f32" if dtype == torch.float32 else "bf16"
            with rec.scope(f"combine-{tag}-{m}", detail=True):
                res = moe.combine(y, wts)
            for name, t in (("y", y), ("wts", wts), ("out", res)):
                dump.save(f"combine-{tag}-{m}", name, t)
    (out / "fixtures.json").write_text(json.dumps({"seed": a.seed, "d": D, "experts": E, "top_k": TOP_K,
                                                   "cases": dump.index}, indent=1) + "\n")
    rec.dump(out / "launches.json")
    (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    print(f"kernels {len(rec.kernels)} cases {len(dump.index)} triton {triton.__version__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
