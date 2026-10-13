#!/usr/bin/env python3
"""The Qwen3.6 MoE Python CUDA engine as the Zig engine's oracle: per-layer dumps, MoE routing, GDN state, tokens.

Run it in the Python engine's container on a copy of its tree (never the tree itself), that copy first on PYTHONPATH:

    PYTHONPATH=<copy>/src python3 capture.py --model <checkpoint dir> --out <dump dir>
    python3 capture.py --compare <dump dir A> <dump dir B>        # two runs hash equal

Prompts (prompts.json): story, code, facts (chat turns of 25 to 35 tokens) and long (about 5,000 tokens of module
source, which prefill splits evenly into two chunks of about 2,520 rows: chunks() divides a prompt into the fewest
4,096-row passes of equal size). For each prompt the engine's own passes are dumped as follows.

    <prompt>/cold/stepSSS/      step SSS of the one-row steps from an empty state (up to 32; the chat prompts
                                are shorter), forced token prompt[SSS], every step dumped
    <prompt>/prefill/           the whole prompt's prefill pass (long: tensor digests only, digests.json, one chunk
                                after the other with call numbers running on)
    <prompt>/state/             the committed state after prefill (see below)
    <prompt>/warm/stepSSS/      32 teacher-forced one-row steps after prefill, forced token: the prompt's first
                                greedy token, then each step's greedy argmax (the serial reply; tokens.json)
    <prompt>/tokens.json        the prompt's hash, forced and sampled tokens, each step's logits sha256, and the
                                engine's own generate(draft=False) tokens over the same count (must equal)

A pass directory holds raw little-endian tensors (shape and dtype in dump.json: name -> [dtype, shape, sha256]):

    NNN_h.bin NNN_y.bin NNN_xs.bin   the NNN-th add_rmsnorm call: its residual sum h (bf16 [rows, hidden]), the
                                     normed rows y (bf16), and the y group sums xs (the matmul's activation sums,
                                     fp32). Calls run in order, two a layer (input norm, then post norm), then the
                                     final norm of the pass: 81 calls with 40 layers
    mNN_logits.bin                   MoE call NN (one a layer, the layer's index in a one-chunk pass): router logits,
                                     fp32 [rows, 257] (256 experts, the shared expert's gate last)
    mNN_pick.bin  mNN_wts.bin        its picks (int32 [rows, 9]: 8 experts by logit, the lower id on a tie, then
                                     the shared expert 256) and weights (fp32 [rows, 9])
    mNN_out.bin                      the layer's MoE output (bf16 [rows, hidden])
    logits.bin                       a decode step's logits (bf16 [1, vocab])

The state after prefill (<prompt>/state/):

    LNN_rec.bin LNN_conv.bin   delta-net layer NN: the recurrent state (fp32 [32, 128, 128] value heads, value rows,
                               key columns) and the conv window (bf16 [3, 8192])
    LNN_k.bin LNN_v.bin        attention layer NN: the cache rows past (bf16 [pos, 2, 256])

weights.txt holds "name sha256 bytes" lines for the engine's loaded weights (tiled 4-bit projections, norms,
conv, router rows, expert stacks), the lines the Zig upload prints for its own device bytes. manifest.json holds
every dump file's sha256 and the tree's; timing.json (outside the manifest) holds wall times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

TEXTS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "facts": "Explain how a refrigerator moves heat out of its cabinet, step by step.",
}
LONG_BODY = 5000          # tokens of module source in the long prompt: two prefill chunks
TEACHER = 32              # teacher-forced one-row steps, cold and warm


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompts(model: Path) -> dict[str, list[int]]:
    import textwrap

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    out = {name: tok.encode(chat(text), add_special_tokens=False).ids for name, text in TEXTS.items()}
    words = Path(textwrap.__file__).read_text()
    body = tok.encode(words * (1 + LONG_BODY // max(1, len(words) // 3)), add_special_tokens=False).ids[:LONG_BODY]
    head = tok.encode("<|im_start|>user\nReview this module and list its public functions:\n\n",
                      add_special_tokens=False).ids
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
    out["long"] = head + body + tail
    return out


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


def raw(t):
    """A tensor's bytes on the host."""

    import torch

    torch.cuda.synchronize()
    return t.detach().contiguous().view(torch.uint8).cpu().numpy()


def digest(t) -> str:
    return hashlib.sha256(raw(t).tobytes()).hexdigest()


class Dumps:
    """Tensors by name under the current pass directory (raw files, or digests only); None turns dumping off."""

    def __init__(self) -> None:
        self.dir: Path | None = None
        self.hash_only = False
        self.norms = 0                      # add_rmsnorm calls this pass
        self.moes = 0                       # moe.run calls this pass
        self.seen: dict[str, list] = {}

    def at(self, d: Path | None, hash_only: bool = False) -> None:
        self.flush()
        self.dir, self.hash_only, self.norms, self.moes, self.seen = d, hash_only, 0, 0, {}
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    def flush(self) -> None:
        if self.dir is not None:
            name = "digests.json" if self.hash_only else "dump.json"
            (self.dir / name).write_text(json.dumps(self.seen, indent=0, sort_keys=True) + "\n")

    def save(self, name: str, t) -> None:
        if self.dir is None:
            return
        data = raw(t)
        self.seen[name] = [str(t.dtype).removeprefix("torch."), list(t.shape), hashlib.sha256(data.tobytes()).hexdigest()]
        if not self.hash_only:
            data.tofile(self.dir / f"{name}.bin")


def hook(dumps: Dumps) -> None:
    """Module-level wrappers (the engine's sources stay as they are): every add_rmsnorm, router, select_rows, run."""

    from tensorfold.cuda import moe
    from tensorfold.families.qwen3_5.cuda import glue

    norm, router, select_rows, run = glue.add_rmsnorm, moe.router, moe.select_rows, moe.run

    def traced_norm(x, r, w, eps):
        h, y, xs = norm(x, r, w, eps)
        if dumps.dir is not None:
            i = dumps.norms
            dumps.norms += 1
            for name, t in (("h", h), ("y", y), ("xs", xs)):
                dumps.save(f"{i:03d}_{name}", t)
        return h, y, xs

    def traced_router(x, rows, out=None):
        logits = router(x, rows, out)
        dumps.save(f"m{dumps.moes:02d}_logits", logits)
        return logits

    def traced_select(logits, buf, top_k, experts):
        select_rows(logits, buf, top_k, experts)
        n = logits.shape[0]
        dumps.save(f"m{dumps.moes:02d}_pick", buf.pick[:n])
        dumps.save(f"m{dumps.moes:02d}_wts", buf.wts[:n])

    def traced_run(x, m, *, prefill=False):
        out = run(x, m, prefill=prefill)
        dumps.save(f"m{dumps.moes:02d}_out", out)
        dumps.moes += 1
        return out

    glue.add_rmsnorm, moe.router, moe.select_rows, moe.run = traced_norm, traced_router, traced_select, traced_run


def weight_lines(w) -> list[str]:
    """'name sha256 bytes' for the loaded weights: the names and bytes the Zig upload reports."""

    out: list[str] = []

    def put(name: str, t) -> None:
        data = raw(t)
        out.append(f"{name} {hashlib.sha256(data.tobytes()).hexdigest()} {data.size}")

    def lin(name: str, q) -> None:
        for part, t in (("weight", q.weight), ("scales", q.scales), ("biases", q.biases)):
            put(f"{name}.{part}", t)

    lin("embed", w.embed)
    put("norm", w.norm)
    lin("head", w.head)
    put("inv_freq", w.inv_freq)
    for i, layer in enumerate(w.layers):
        p = f"L{i:02d}."
        put(p + "input_norm", layer.input_norm)
        put(p + "post_norm", layer.post_norm)
        if layer.gdn is not None:
            g = layer.gdn
            for n in ("qkv", "z", "b", "a", "out"):
                lin(p + "gdn." + n, getattr(g, n))
            for n in ("conv", "A_log", "dt_bias", "norm"):
                put(p + "gdn." + n, getattr(g, n))
        else:
            a = layer.attn
            for n in ("q", "k", "v", "o"):
                lin(p + "attn." + n, getattr(a, n))
            put(p + "attn.q_norm", a.q_norm)
            put(p + "attn.k_norm", a.k_norm)
        put(p + "moe.router", layer.moe.router)
        put(p + "moe.up", layer.moe.experts.up)
        put(p + "moe.down", layer.moe.experts.down)
    return out


def save_state(dumps: Dumps, w, st) -> None:
    """The committed state: every delta-net layer's recurrence and conv window, every attention layer's cache rows."""

    for i, layer in enumerate(w.layers):
        if layer.linear:
            dumps.save(f"L{i:02d}_rec", st.rec[i])
            dumps.save(f"L{i:02d}_conv", st.conv[i])
        else:
            k, v = st.kv[i]
            dumps.save(f"L{i:02d}_k", k[:st.pos])
            dumps.save(f"L{i:02d}_v", v[:st.pos])


def steps(w, dumps: Dumps, root: Path, st, tokens: list[int] | None = None, first: int | None = None,
          count: int = TEACHER) -> tuple[list[int], list[int], list[str]]:
    """``count`` one-row steps, forcing ``tokens`` or ``first`` then each greedy token: (forced, sampled, shas)."""

    import torch
    from tensorfold.cuda.sampling import sample_rows
    from tensorfold.families.qwen3_5.cuda.forward import commit, tree_forward

    forced, sampled, logit_sha = [], [], []
    for i in range(count):
        t = tokens[i] if tokens is not None else first if i == 0 else sampled[-1]
        dumps.at(root / f"step{i:03d}")
        logits, record = tree_forward(w, torch.tensor([t], dtype=torch.int32, device=w.norm.device), [-1], st)
        dumps.save("logits", logits)
        forced.append(t)
        sampled.append(sample_rows(logits, [st.pos + 1], None)[0])
        logit_sha.append(digest(logits))
        commit(st, record, [0])
    dumps.at(None)
    return forced, sampled, logit_sha


def manifest(out: Path) -> dict:
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "timing.json"):
            files[p.relative_to(out).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    tree = hashlib.sha256("".join(f"{k} {v}\n" for k, v in files.items()).encode()).hexdigest()
    return {"tree": tree, "files": files}


def compare(a: Path, b: Path) -> int:
    ma, mb = json.loads((a / "manifest.json").read_text()), json.loads((b / "manifest.json").read_text())
    bad = sorted(k for k in ma["files"].keys() | mb["files"].keys() if ma["files"].get(k) != mb["files"].get(k))
    print(f"{len(ma['files'])} files, tree {ma['tree']} against {mb['tree']}: "
          f"{'EQUAL' if not bad else str(len(bad)) + ' differ: ' + ', '.join(bad[:8])}")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar="DIR")
    ap.add_argument("--weights-only", action="store_true", help="write weights.txt and stop")
    ap.add_argument("--context", type=int, default=8192)
    a = ap.parse_args()
    if a.compare:
        return compare(Path(a.compare[0]), Path(a.compare[1]))
    if not a.model or not a.out:
        ap.error("--model and --out are required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Path(a.model)

    import torch
    import tensorfold
    from tensorfold.families.qwen3_5.cuda.decode import prefill
    from tensorfold.families.qwen3_5.cuda.forward import State
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    ids = prompts(model)
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    dumps = Dumps()
    hook(dumps)
    t0 = time.perf_counter()
    eng = Qwen36Engine(model, depth=0, context=a.context, context_explicit=True, streams=1)
    w = eng.w
    timing = {"startup_s": round(time.perf_counter() - t0, 2)}
    info = {"tensorfold": str(Path(tensorfold.__file__).parent), "version": getattr(tensorfold, "__version__", None),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "context": eng.context_window,
            "eos": list(eng.eos), "prompt_rows": w.prompt_rows, "layers": len(w.layers),
            "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "prompt_lengths": {k: len(v) for k, v in ids.items()}}
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)
    (out / "weights.txt").write_text("\n".join(weight_lines(w)) + "\n")
    print("weights.txt written", flush=True)
    if a.weights_only:
        return 0

    for name, prompt in ids.items():
        d = out / name
        t1 = time.perf_counter()
        cold_forced, cold, cold_sha = steps(w, dumps, d / "cold", State(w), tokens=prompt[:TEACHER],
                                            count=min(TEACHER, len(prompt)))
        dumps.at(d / "prefill", hash_only=name == "long")
        st, first = prefill(w, prompt, None, limit=eng.context_window)
        dumps.at(d / "state")
        save_state(dumps, w, st)
        dumps.at(None)
        forced, sampled, sha = steps(w, dumps, d / "warm", st, first=first)
        check: list[int] = []
        eng.generate(prompt, TEACHER + 1, None, lambda new: check.extend(map(int, new)) and False, draft=False,
                     stop_eos=False)
        tokens = {"prompt_sha": sha12(prompt), "prompt_rows": len(prompt), "first": first,
                  "cold_forced": cold_forced, "cold_sampled": cold, "cold_logits_sha256": cold_sha,
                  "warm_forced": forced, "warm_sampled": sampled, "warm_logits_sha256": sha,
                  "generate": check, "generate_equal": check == [first] + sampled}
        (d / "tokens.json").write_text(json.dumps(tokens, indent=1) + "\n")
        timing[name] = round(time.perf_counter() - t1, 2)
        print(f"{name}: {len(prompt)} rows, first {first}, reply sha {sha12(sampled)}, "
              f"generate equal {tokens['generate_equal']}", flush=True)

    (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    m = manifest(out)
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"done: {len(m['files'])} files, tree {m['tree']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
