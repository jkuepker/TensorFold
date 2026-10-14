#!/usr/bin/env python3
"""The Qwen3.6 CUDA engine's own router launches: real weights and activations dumped as capture_router.py fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

DUMP_ROWS = 512             # calls past this many rows are recorded but not dumped
EVERY = 9                   # beyond a row count's first call, every EVERY-th one is dumped
LIMIT = 80


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="folder of triton_aot_manifest.py")
    a = ap.parse_args()
    sys.path.insert(0, a.tools)
    import triton_aot_manifest as aot

    rec = aot.Recorder().install()
    import torch

    from tensorfold.cuda import moe
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine
    from tokenizers import Tokenizer

    out = Path(a.out)
    root = out / "fixtures"
    seen: dict[int, int] = {}
    index: dict[str, dict] = {}
    state = {"calls": 0, "case": None, "tag": None}

    def save(case: str, name: str, t) -> None:
        d = root / case
        d.mkdir(parents=True, exist_ok=True)
        torch.cuda.synchronize()
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tofile(d / f"{name}.bin")
        index.setdefault(case, {})[name] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}

    router, select_rows, combine = moe.router, moe.select_rows, moe.combine

    def traced_router(x, rows, o=None):
        res = router(x, rows, o)
        m = x.shape[0]
        state["calls"] += 1
        n = seen.get(m, 0)
        seen[m] = n + 1
        take = (m <= DUMP_ROWS and (n == 0 or n % EVERY == 0) and len(index) < LIMIT
                and not torch.cuda.is_current_stream_capturing())
        state["tag"] = f"{m}-c{state['calls']}" if take else None
        if take:
            save(f"router-{state['tag']}", "x", x)
            save(f"router-{state['tag']}", "rows", rows)
            save(f"router-{state['tag']}", "logits", res)
        return res

    def traced_select(logits, buf, top_k, experts):
        select_rows(logits, buf, top_k, experts)
        tag = state["tag"]
        if tag is not None:
            m = logits.shape[0]
            for case in (f"topk-{tag}", f"router-{tag}"):
                save(case, "logits", logits)
                save(case, "pick", buf.pick[:m])
                save(case, "wts", buf.wts[:m])

    def traced_combine(y, wts):
        res = combine(y, wts)
        tag = state["tag"]
        if tag is not None:
            form = "bf16" if y.dtype == torch.bfloat16 else "f32"
            case = f"combine-{form}-{tag}"
            save(case, "y", y)
            save(case, "wts", wts)
            save(case, "out", res)
            state["tag"] = None
        return res

    moe.router, moe.select_rows, moe.combine = traced_router, traced_select, traced_combine
    model = Path(a.model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    body = Path(textwrap.__file__).read_text()
    texts = {"short": "Name three colors.", "medium": body[:1500], "long": body[:6000]}
    prompts = {k: tok.encode(f"<|im_start|>user\n{v}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                             add_special_tokens=False).ids for k, v in texts.items()}
    eng = Qwen36Engine(model, context=8192, context_explicit=True)
    results = {}
    for draft in (False, True):
        for name, ids in prompts.items():
            toks: list[int] = []
            with rec.scope(f"{'drafted' if draft else 'serial'}-{name}"):
                eng.generate(ids, 48, None, lambda new: toks.extend(map(int, new)) and False, draft=draft)
            results[f"{'drafted' if draft else 'serial'}/{name}"] = {"prompt_tokens": len(ids), "tokens": toks}
            print(name, draft, len(ids), len(toks), flush=True)
    moe.router, moe.select_rows, moe.combine = router, select_rows, combine
    cfg = eng.w.config
    (out / "fixtures.json").write_text(json.dumps({"d": int(cfg.hidden), "experts": int(cfg.experts),
                                                   "top_k": int(cfg.top_k), "cases": index}, indent=1) + "\n")
    (out / "results.json").write_text(json.dumps(results) + "\n")
    rec.dump(out / "launches.json")
    print(f"kernels {len(rec.kernels)} cases {len(index)} calls {state['calls']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
