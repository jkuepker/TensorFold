"""Qwen3.6 concurrent drafted rounds, Python engine's side (capture.py --lanes N): Triton launches of shared rounds."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

SAMPLED = {"seed": 20240601, "temperature": 0.8, "top_k": 40, "top_p": 0.9}
TOKENS = 96


def run(a, out: Path, rec) -> int:
    import capture as cap
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    model = Path(a.model)
    ids = cap.prompts(model)
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    t0 = time.perf_counter()
    eng = Qwen36Engine(model, context=a.context, context_explicit=True, streams=a.lanes)
    print(f"engine with {a.lanes} streams up in {time.perf_counter() - t0:.0f}s", flush=True)
    names = list(ids)

    def together(n: int, drafted: bool, sampled: bool) -> list[list[int]]:
        got: list[list[int]] = [[] for _ in range(n)]

        def one(i: int) -> None:
            sampling = Sampling(**SAMPLED) if sampled else None
            prompt = ids[names[i % len(names)]]
            eng.generate(prompt, TOKENS, sampling, lambda new, i=i: got[i].extend(map(int, new)) and False, draft=drafted,
                         stop_eos=False)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return got

    results = {}
    for n in sorted({2, 3, 4, a.lanes // 2 or 1, a.lanes}):
        for drafted in (True, False):
            for sampled in (False, True):
                key = f"{n}_{'drafted' if drafted else 'plain'}_{'sampled' if sampled else 'greedy'}"
                t1 = time.perf_counter()
                results[key] = together(n, drafted, sampled)
                print(f"{key}: {[len(r) for r in results[key]]} tokens in {time.perf_counter() - t1:.0f}s", flush=True)
    (out / "tokens.json").write_text(json.dumps({"names": names, "tokens": TOKENS, "results": results}) + "\n")
    eng.close()
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(cap.specialization(), indent=1, sort_keys=True) + "\n")
    print("done", flush=True)
    return 0
