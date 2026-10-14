"""The Qwen3.6 MTP head and its drafted rounds on the Python CUDA engine, as the Zig oracle (capture.py --mtp)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

DEPTH, CONFIDENCE = 3, 0.3
COUNT = {"story": 96, "code": 96, "facts": 96, "long": 48}
SAMPLED = {"seed": 20240601, "temperature": 0.8, "top_k": 40, "top_p": 0.9}
MAX_VERIFY_WIDTH = 16


def mtp_lines(eng, raw) -> list[str]:
    """'name sha256 bytes' for the head's tensors: what the Zig MTP upload reports for its own device bytes."""

    out: list[str] = []

    def put(name: str, t) -> None:
        data = raw(t)
        out.append(f"{name} {hashlib.sha256(data.tobytes()).hexdigest()} {data.size}")

    def lin(name: str, q) -> None:
        for part, t in (("weight", q.weight), ("scales", q.scales), ("biases", q.biases)):
            put(f"{name}.{part}", t)

    m = eng.head.m
    put("mtp.norm_e", m.norm_e)
    put("mtp.norm_h", m.norm_h)
    lin("mtp.fc_e", m.fc_e)
    lin("mtp.fc_h", m.fc_h)
    put("mtp.input_norm", m.input_norm)
    put("mtp.post_norm", m.post_norm)
    for n in ("q", "k", "v", "o"):
        lin("mtp.attn." + n, getattr(m.attn, n))
    put("mtp.attn.q_norm", m.attn.q_norm)
    put("mtp.attn.k_norm", m.attn.k_norm)
    put("mtp.moe.router", m.moe.router)
    put("mtp.moe.up", m.moe.experts.up)
    put("mtp.moe.down", m.moe.experts.down)
    put("mtp.norm", m.norm)
    lin("mtp.head", eng.head.head)
    put("mtp.ids", eng.head.ids)
    return out


class Trace:
    """Wrappers on the head, the rounds and the verify pass, recording what the Zig engine must reproduce."""

    def __init__(self, cap, dumps, eng, decode, mtp_mod) -> None:
        self.cap, self.dumps, self.eng, self.decode, self.mtp_mod = cap, dumps, eng, decode, mtp_mod
        self.root: Path | None = None
        self.hash_only = False
        self.calls: list[dict] = []
        self.rounds: list[dict] = []
        self.guesses: list[dict] = []
        self.widths: set[int] = set()
        self.last = ""
        self.phase = "prefill"
        self.verify_log: list[dict] = []
        head = mtp_mod.Head
        self.forward, self.logits = head.forward, head.logits
        self.round, self.draft, self.tree_forward = decode.mtp_round, decode.draft, decode.tree_forward
        me = self

        def forward(self_, cache, states, tokens, p0, staged=None):
            if me.root is None:
                return me.forward(self_, cache, states, tokens, p0, staged)
            i = len(me.calls)
            d = me.root / f"c{i:03d}"
            me.dumps.at(d, hash_only=me.hash_only)
            me.raw(d, "states", states)
            out = me.forward(self_, cache, states, tokens, p0, staged)
            me.raw(d, "out", out)
            me.last = f"c{i:03d}"
            me.calls.append({"i": i, "dir": me.last, "phase": me.phase, "n": int(states.shape[0]), "p0": int(p0),
                             "tokens": [int(t) for t in tokens], "logits": False})
            return out

        def logits(self_, normed):
            out = me.logits(self_, normed)
            if me.root is not None and me.calls and not me.calls[-1]["logits"]:
                me.raw(me.root / me.last, "logits", out)
                me.calls[-1]["logits"] = True
            return out

        def draft(logits_, position, sampling, ids=None):
            token, prob = me.draft(logits_, position, sampling, ids)
            if me.root is not None:
                me.guesses.append({"position": int(position), "token": int(token), "prob": float(prob),
                                   "call": len(me.calls) - 1})
            return token, prob

        def mtp_round(st, mc, carry, pending, room, sampling, context, copies, **kw):
            if me.root is None:
                return me.round(st, mc, carry, pending, room, sampling, context, copies, **kw)
            first, pos = len(me.guesses), int(st.pos)
            calls = len(me.calls)
            tokens, path, new, nxt = me.round(st, mc, carry, pending, room, sampling, context, copies, **kw)
            me.rounds.append({"pos": pos, "pending": int(pending), "room": int(room), "tokens": [int(t) for t in tokens],
                              "path": [int(p) for p in path], "new": [int(t) for t in new],
                              "carry": len(carry.tokens), "kind": "head" if len(me.guesses) > first else "copy",
                              "guesses": me.guesses[first:], "calls": [calls, len(me.calls)]})
            return tokens, path, new, nxt

        def tree_forward(w, ids, parents, st, **kw):
            n = int(ids.shape[0])
            if me.root is None:
                return me.tree_forward(w, ids, parents, st, **kw)
            full = 2 <= n <= MAX_VERIFY_WIDTH and n not in me.widths
            if full:
                me.widths.add(n)
            me.dumps.at(me.root / "verify" / f"w{n:02d}" if full else None)
            out = me.tree_forward(w, ids, parents, st, **kw)
            me.dumps.save("logits", out[0])
            me.dumps.save("states", out[2])
            me.verify_log.append({"n": n, "pos": int(st.pos), "tokens": [int(t) for t in ids.tolist()],
                                  "logits_sha256": [me.cap.digest(out[0][r]) for r in range(n)]})
            me.dumps.at(None)
            return out

        head.forward, head.logits = forward, logits
        decode.draft, decode.mtp_round, decode.tree_forward = draft, mtp_round, tree_forward

    def raw(self, d: Path, name: str, t) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.cap.raw(t).tofile(d / f"{name}.bin")

    def start(self, root: Path | None, hash_only: bool = False) -> None:
        self.root, self.hash_only = root, hash_only
        self.calls, self.rounds, self.guesses, self.widths, self.verify_log, self.phase = [], [], [], set(), [], "prefill"
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        self.dumps.at(None)
        if self.root is not None:
            (self.root / "calls.json").write_text(json.dumps(self.calls) + "\n")
            (self.root / "rounds.json").write_text(json.dumps(self.rounds, indent=0) + "\n")
            (self.root / "verify.json").write_text(json.dumps(self.verify_log) + "\n")
        self.root = None


def sweep(w, eng, prompt, decode, mtp_mod, cap) -> None:
    """Every verify width and head row count the engine can launch, so the recorded Triton set covers them all."""

    import torch

    sampling = None
    st, mc, first, carry = decode.prefill(w, eng.head, prompt, sampling)
    dev = w.norm.device
    for n in range(1, MAX_VERIFY_WIDTH + 1):
        tokens = [int(first)] * n
        decode.tree_forward(w, torch.tensor(tokens, dtype=torch.int32, device=dev), list(range(-1, n - 1)), st, hidden=True)
        states = torch.zeros((n, w.config.hidden), dtype=torch.bfloat16, device=dev)
        eng.head.forward(mc.view(len(prompt) + 64), states, tokens, mc.pos)
    torch.cuda.synchronize()


def run(a, out: Path, rec) -> int:
    """capture.py --mtp: the head's calls, rounds and verify passes of every prompt, greedy and one seeded request."""

    import capture as cap
    import torch
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen3_5_moe.cuda import decode, mtp as mtp_mod
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    model = Path(a.model)
    ids = cap.prompts(model)
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    dumps = cap.Dumps()
    cap.hook(dumps)
    t0 = time.perf_counter()
    eng = Qwen36Engine(model, depth=DEPTH, confidence=CONFIDENCE, context=a.context, context_explicit=True, streams=1)
    w = eng.w
    (out / "mtp_weights.txt").write_text("\n".join(mtp_lines(eng, cap.raw)) + "\n")
    info = {"startup_s": round(time.perf_counter() - t0, 2), "depth": DEPTH, "confidence": CONFIDENCE,
            "draft_ids": int(eng.head.ids.shape[0]), "sampled": SAMPLED, "counts": COUNT,
            "gpu": torch.cuda.get_device_name(0)}
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)
    trace = Trace(cap, dumps, eng, decode, mtp_mod)

    def generated(prompt, count, sampling, drafted):
        eng.cache.entries = []                        # a kept prompt state would change what the prefill does
        got: list[int] = []
        stats = eng.generate(prompt, count, sampling, lambda new: got.extend(map(int, new)) and False, draft=drafted,
                             stop_eos=False)
        return got, {k: stats.get(k) for k in ("drafted", "accepted", "rounds", "min_rows")}

    def eager(prompt, count, sampling, root, hash_only):
        trace.start(root, hash_only)
        st, mc, first, carry = decode.prefill(w, eng.head, prompt, sampling)
        trace.phase = "decode"
        res = decode.mtp_decode(w, eng.head, st, mc, carry, first, count, sampling, depth=DEPTH, confidence=CONFIDENCE,
                                stop_eos=False, prompt=prompt)
        trace.stop()
        return res

    for name, prompt in ids.items():
        t1 = time.perf_counter()
        count = COUNT[name]
        for kind, sampling in (("greedy", None), ("sampled", Sampling(**SAMPLED))):
            root = out / name / kind
            res = eager(prompt, count, sampling, root if kind == "greedy" else None, name == "long")
            graphs, graph_stats = generated(prompt, count, sampling, True)
            serial, _ = generated(prompt, count, sampling, False)
            rounds = res.rounds
            tokens = {"prompt_sha": cap.sha12(prompt), "prompt_rows": len(prompt), "count": count,
                      "eager": [int(t) for t in res.tokens], "graphs": graphs, "serial": serial,
                      "eager_stats": {"drafted": res.drafted, "accepted": res.accepted, "rounds": rounds,
                                      "widths": res.widths},
                      "graph_stats": graph_stats,
                      "eager_equals_serial": [int(t) for t in res.tokens] == serial,
                      "graphs_equal_serial": graphs == serial}
            (out / name / kind).mkdir(parents=True, exist_ok=True)
            (out / name / kind / "tokens.json").write_text(json.dumps(tokens, indent=1) + "\n")
            print(f"{name} {kind}: {len(prompt)} rows, drafted {res.drafted} accepted {res.accepted} in {rounds} rounds, "
                  f"eager==serial {tokens['eager_equals_serial']} graphs==serial {tokens['graphs_equal_serial']} "
                  f"({time.perf_counter() - t1:.0f}s)", flush=True)
    sweep(w, eng, ids["story"], decode, mtp_mod, cap)
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(cap.specialization(), indent=1, sort_keys=True) + "\n")
    m = cap.manifest(out)
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"done: {len(m['files'])} files, tree {m['tree']}", flush=True)
    return 0
