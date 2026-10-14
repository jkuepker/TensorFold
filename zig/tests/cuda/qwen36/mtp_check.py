#!/usr/bin/env python3
"""Qwen3.6 MTP drafts through the native CUDA server against "draft": false and the Python engine (stdlib only)."""

from __future__ import annotations

import json
import sys
import time
import urllib.request

import serve_check as sc

SAMPLED = {"seed": 20240601, "temperature": 0.8, "top_k": 40, "top_p": 0.9}


def ask(url: str, body: dict) -> dict:
    t0 = time.time()
    reply = sc.post(url, "/v1/chat/completions" if "messages" in body else "/v1/completions", body)
    run = reply.get("tensorfold") or {}
    return {"ids": run.get("token_ids"), "spec": reply.get("speculative") or {}, "tps": run.get("tokens_per_second"), "seconds": time.time() - t0,
            "run": run}


def rate(spec: dict) -> str:
    d, a = spec.get("drafted"), spec.get("accepted")
    return f"{a}/{d} ({a / d:.2f}) in {spec.get('rounds')} rounds" if d else f"no drafts ({spec})"


def main(url: str, fixture_path: str, oracle: str) -> int:
    failed = 0
    fixture = json.load(open(fixture_path))
    print(f"-- chat fixture, greedy, {fixture['engine']}")
    for r in fixture["requests"]:
        if r.get("ids_only"):
            continue
        body = {"messages": r["messages"], "max_tokens": sc.MAX_TOKENS, "temperature": 0, "return_token_ids": True,
                "chat_template_kwargs": r["chat_template_kwargs"]}
        if r.get("tools"):
            body["tools"] = r["tools"]
        on, off = ask(url, body), ask(url, dict(body, draft=False))
        same_off, same_py = on["ids"] == off["ids"], on["ids"] == r["reply_ids"]
        failed += (not same_off) + (not same_py)
        print(f"{r['name']}: drafted == draft:false {same_off}, == python serial {same_py}; accepted {rate(on['spec'])}; "
              f"{on['tps'] and round(on['tps'], 1)} tok/s drafted, {off['tps'] and round(off['tps'], 1)} tok/s off", flush=True)
    prompts = json.load(open(f"{oracle}/prompts.json"))
    for kind in ("greedy", "sampled"):
        print(f"-- oracle prompts, {kind}")
        for name, ids in prompts.items():
            want = json.load(open(f"{oracle}/{name}/{kind}/tokens.json"))
            body = {"prompt": ids, "max_tokens": want["count"], "return_token_ids": True, "ignore_eos": True,
                    "temperature": 0, "top_k": 0, "top_p": 1.0}
            if kind == "sampled":
                body.update(SAMPLED)
            on, off = ask(url, body), ask(url, dict(body, draft=False))
            same_off, same_py = on["ids"] == off["ids"], on["ids"] == want["serial"]
            failed += (not same_off) + (not same_py)
            py = want["eager_stats"]
            print(f"{name}: drafted == draft:false {same_off}, == python serial {same_py}; native {rate(on['spec'])}, "
                  f"python {py['accepted']}/{py['drafted']} ({py['accepted'] / py['drafted']:.2f}) in {py['rounds']} rounds; "
                  f"{on['tps'] and round(on['tps'], 1)} tok/s drafted, {off['tps'] and round(off['tps'], 1)} tok/s off", flush=True)
            if not same_py:
                print("   " + sc.first_difference(on["ids"] or [], want["serial"]))
    print("PASS" if not failed else f"FAIL: {failed} differences")
    return 1 if failed else 0


USAGE = (
    "Qwen3.6 MTP drafts through the native CUDA server against \"draft\": false and the Python engine.\n"
    "\n"
    "  mtp_check.py URL FIXTURE ORACLE\n"
    "      FIXTURE: serve_check.py's; ORACLE: capture.py --mtp's folder (prompts.json, <prompt>/<kind>/tokens.json)\n"
    "\n"
    "For the fixture's chat requests (greedy, thinking off) and for the oracle's four prompts (greedy, and one seeded\n"
    "sampled request each, sent as token ids): the drafted reply must equal the same request with\n"
    "\"draft\": false token for token, and the Python engine's serial reply. Acceptance per request is printed\n"
    "beside the Python engine's own on the same prompt.\n"
)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(USAGE)
    sys.exit(main(*sys.argv[1:]))
