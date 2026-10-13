#!/usr/bin/env python3
"""Checks for a converted MiniCPM-V 4.7 MLX checkpoint (see minicpmv47_to_mlx.py).

  inventory  diff the language-model tensors (name, shape, dtype, bits per tensor) against a reference inventory
             JSON of mlx-community/Qwen3.6-35B-A3B-4bit ({name: {dtype, shape}}, built from its safetensors headers);
             only vocab-sized dimensions (248144 here, 248320 there) may differ.
  stats      min/mean/max of some norms, to compare against the reference's.
  dequant    dequantize some tensors and compare them to the BF16 source (relative L2 error).

Run with the Python that has mlx-lm.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def inventory(out: Path) -> dict:
    inv = {}
    for p in sorted(out.glob("*.safetensors")):
        for k, v in header(p).items():
            inv[k] = {"dtype": v["dtype"], "shape": v["shape"]}
    return inv


def cmd_inventory(a) -> None:
    mine, ref = inventory(a.out), json.load(open(a.ref))
    mine_lm = {k: v for k, v in mine.items() if k.startswith("language_model.")}
    ref_lm = {k: v for k, v in ref.items() if k.startswith("language_model.")}
    print(f"language-model tensors: output {len(mine_lm)}, reference {len(ref_lm)}")
    print("only in output:", sorted(set(mine_lm) - set(ref_lm))[:10])
    print("only in reference:", sorted(set(ref_lm) - set(mine_lm))[:10])
    diffs = []
    for k in sorted(set(mine_lm) & set(ref_lm)):
        m, r = mine_lm[k], ref_lm[k]
        if m["dtype"] != r["dtype"] or m["shape"] != r["shape"]:
            diffs.append((k, m, r))
    vocab_only = [d for d in diffs if d[1]["dtype"] == d[2]["dtype"] and len(d[1]["shape"]) == len(d[2]["shape"])
                  and [i for i, (x, y) in enumerate(zip(d[1]["shape"], d[2]["shape"])) if x != y] == [0]
                  and d[1]["shape"][0] == a.vocab and d[2]["shape"][0] == a.ref_vocab]
    other = [d for d in diffs if d not in vocab_only]
    print(f"shape/dtype differences: {len(diffs)} (vocab-only {len(vocab_only)}, other {len(other)})")
    for k, m, r in vocab_only:
        print("  vocab", k, m["shape"], "vs", r["shape"])
    for k, m, r in other:
        print("  OTHER", k, m, "vs", r)
    # bits per tensor from packed width: weight cols * 32 / (scales cols * group)
    cfg = json.load(open(a.out / "config.json"))
    q = cfg["quantization"]
    bad = 0
    for k, v in mine_lm.items():
        if not k.endswith(".weight") or v["dtype"] != "U32":
            continue
        base = k[: -len(".weight")]
        s = mine_lm[base + ".scales"]["shape"]
        w = v["shape"]
        gs = (q.get(base) or q)["group_size"]
        bits = (q.get(base) or q)["bits"]
        if w[-1] * 32 != s[-1] * gs * bits:
            bad += 1
            print("  BITS MISMATCH", k, w, s, bits, gs)
    eights = sorted(k for k, v in q.items() if isinstance(v, dict) and v["bits"] == 8)
    print(f"config bits: default {q['bits']}/g{q['group_size']}, {len(eights)} per-tensor 8-bit overrides, packed-width mismatches {bad}")
    ref_cfg = json.load(open(a.ref_config))["quantization"]
    print("quantization block equal to reference:", q == ref_cfg)
    print("quantization_config equal to quantization:", cfg["quantization_config"] == q)
    vis = {k: v for k, v in mine.items() if not k.startswith("language_model.")}
    kinds = {(v["dtype"]) for v in vis.values()}
    print(f"non-language tensors: {len(vis)} dtypes {kinds}; vision_tower {sum(k.startswith('model.vision_tower.') for k in vis)}, merger {sum(k.startswith('model.merger.') for k in vis)}")


def cmd_stats(a) -> None:
    import mlx.core as mx

    idx = json.loads((a.out / "model.safetensors.index.json").read_text())["weight_map"]
    for k in a.names:
        v = mx.load(str(a.out / idx[k]))[k].astype(mx.float32)
        print(f"{k} {v.dtype} {tuple(v.shape)} min {v.min().item():.4f} mean {v.mean().item():.4f} max {v.max().item():.4f}")


def cmd_dequant(a) -> None:
    import mlx.core as mx

    cfg = json.loads((a.out / "config.json").read_text())["quantization"]
    idx = json.loads((a.out / "model.safetensors.index.json").read_text())["weight_map"]
    sidx = json.loads((a.src / "model.safetensors.index.json").read_text())["weight_map"]
    cache = {}

    def load(root, imap, k):
        f = imap[k]
        if (root, f) not in cache:
            cache.clear()
            cache[(root, f)] = mx.load(str(root / f))
        return cache[(root, f)][k]

    def check(name, base, src_name, sl=None, shift=0.0):
        spec = cfg.get(base, cfg)
        w, s, b = (load(a.out, idx, base + x) for x in (".weight", ".scales", ".biases"))
        if sl is not None:
            w, s, b = w[sl], s[sl], b[sl]
        deq = mx.dequantize(w, s, b, group_size=spec["group_size"], bits=spec["bits"]).astype(mx.float32)
        src = load(a.src, sidx, src_name)
        src = (src[sl] if sl is not None else src).astype(mx.float32)
        err = (mx.linalg.norm(deq - src) / mx.linalg.norm(src)).item()
        print(f"{name}: bits {spec['bits']} rel-L2 error {err:.4f}  (shape {tuple(deq.shape)})")

    L = "model.language_model."
    check("embed_tokens", "language_model.model.embed_tokens", L + "embed_tokens.weight")
    check("lm_head", "language_model.lm_head", "lm_head.weight")
    check("layer0 in_proj_qkv", "language_model.model.layers.0.linear_attn.in_proj_qkv", L + "layers.0.linear_attn.in_proj_qkv.weight")
    check("layer3 q_proj", "language_model.model.layers.3.self_attn.q_proj", L + "layers.3.self_attn.q_proj.weight")
    check("layer5 router gate (8-bit)", "language_model.model.layers.5.mlp.gate", L + "layers.5.mlp.gate.weight")
    check("layer5 shared_expert_gate (8-bit)", "language_model.model.layers.5.mlp.shared_expert_gate", L + "layers.5.mlp.shared_expert_gate.weight")
    check("layer5 shared_expert up", "language_model.model.layers.5.mlp.shared_expert.up_proj", L + "layers.5.mlp.shared_expert.up_proj.weight")
    # fused source: gate_up [E, 2*I, H] -> gate = first half rows, up = second half; down [E, H, I]
    for e in (0, 117, 255):
        base = "language_model.model.layers.7.mlp.switch_mlp."
        gu = load(a.src, sidx, L + "layers.7.mlp.experts.gate_up_proj")[e].astype(mx.float32)
        dn = load(a.src, sidx, L + "layers.7.mlp.experts.down_proj")[e].astype(mx.float32)
        for nm, ref in (("gate_proj", gu[:512]), ("up_proj", gu[512:]), ("down_proj", dn)):
            w, s, b = (load(a.out, idx, base + nm + x)[e] for x in (".weight", ".scales", ".biases"))
            deq = mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32)
            err = (mx.linalg.norm(deq - ref) / mx.linalg.norm(ref)).item()
            print(f"layer7 expert {e} {nm}: rel-L2 error {err:.4f}  (shape {tuple(deq.shape)})")
    # norm convention: output == source + 1 for the shifted norms, == source for linear_attn.norm
    for nm, shifted in (("layers.39.input_layernorm.weight", True), ("layers.3.self_attn.q_norm.weight", True),
                        ("norm.weight", True), ("layers.0.linear_attn.norm.weight", False)):
        o = load(a.out, idx, "language_model.model." + nm).astype(mx.float32)
        s = load(a.src, sidx, L + nm).astype(mx.float32)
        d = (o - s).astype(mx.float32)
        print(f"{nm}: output - source in [{d.min().item():.4f}, {d.max().item():.4f}] (expected {'about +1' if shifted else '0'})")
    o = load(a.out, idx, "model.vision_tower.encoder.layers.0.mlp.fc1.weight")
    s = load(a.src, sidx, "model.vision_tower.encoder.layers.0.mlp.fc1.weight")
    print("vision fc1 identical to source:", bool(mx.array_equal(o, s).item()), o.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("out", type=Path)
    p.add_argument("--ref", required=True, help="reference inventory JSON")
    p.add_argument("--ref-config", required=True, help="reference config.json")
    p.add_argument("--vocab", type=int, default=248144)
    p.add_argument("--ref-vocab", type=int, default=248320)
    p.set_defaults(fn=cmd_inventory)
    p = sub.add_parser("stats")
    p.add_argument("out", type=Path)
    p.add_argument("names", nargs="+")
    p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("dequant")
    p.add_argument("out", type=Path)
    p.add_argument("--src", type=Path, required=True)
    p.set_defaults(fn=cmd_dequant)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
