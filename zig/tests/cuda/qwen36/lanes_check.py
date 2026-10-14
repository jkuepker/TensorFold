#!/usr/bin/env python3
"""Qwen3.6 concurrent streams on the native CUDA server: each stream equals the same request alone (stdlib only)."""

from __future__ import annotations

import json
import sys
import threading
import time

import serve_check as sc

SAMPLED = {"seed": 20240601, "temperature": 0.8, "top_k": 40, "top_p": 0.9}


def requests(fixture_path: str, oracle: str) -> list[dict]:
    """Every request: name, endpoint body, and the Python serial reply ids when the oracle has them."""
    out = []
    for r in json.load(open(fixture_path))["requests"]:
        if r.get("ids_only"):
            continue
        body = {"messages": r["messages"], "max_tokens": sc.MAX_TOKENS, "temperature": 0, "return_token_ids": True,
                "chat_template_kwargs": r["chat_template_kwargs"]}
        if r.get("tools"):
            body["tools"] = r["tools"]
        out.append({"name": r["name"], "body": body, "python": r["reply_ids"]})
    for name, ids in json.load(open(f"{oracle}/prompts.json")).items():
        want = json.load(open(f"{oracle}/{name}/greedy/tokens.json"))
        body = {"prompt": ids, "max_tokens": want["count"], "return_token_ids": True, "ignore_eos": True,
                "temperature": 0, "top_k": 0, "top_p": 1.0}
        out.append({"name": "oracle_" + name, "body": body, "python": want["serial"]})
    name = "story"
    want = json.load(open(f"{oracle}/{name}/sampled/tokens.json"))
    ids = json.load(open(f"{oracle}/prompts.json"))[name]
    body = {"prompt": ids, "max_tokens": want["count"], "return_token_ids": True, "ignore_eos": True, "temperature": 0,
            "top_k": 0, "top_p": 1.0, **SAMPLED}
    out.append({"name": "oracle_story_sampled", "body": body, "python": want["serial"]})
    return out


def ask(url: str, body: dict) -> list[int]:
    reply = sc.post(url, "/v1/chat/completions" if "messages" in body else "/v1/completions", body)
    return (reply.get("tensorfold") or {}).get("token_ids") or []


def together(url: str, reqs: list[dict], mixed: bool = False) -> list[list[int]]:
    """All of `reqs` sent at the same moment; each reply's ids, in order."""
    got: list = [None] * len(reqs)
    go = threading.Barrier(len(reqs))

    def one(i: int) -> None:
        go.wait()
        try:
            got[i] = ask(url, dict(reqs[i]["body"], draft=False) if mixed and i % 2 else reqs[i]["body"])
        except Exception as exc:  # noqa: BLE001
            got[i] = exc

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(reqs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return got


def main(url: str, fixture: str, oracle: str, counts: str = "1,4,8") -> int:
    reqs = requests(fixture, oracle)
    solo = {}
    failed = 0
    print(f"-- solo, {len(reqs)} requests")
    for r in reqs:
        solo[r["name"]] = ask(url, r["body"])
        same = r["body"].get("seed") is not None or solo[r["name"]] == r["python"]
        failed += not same
        print(f"{r['name']}: {len(solo[r['name']])} ids, == python serial {same}", flush=True)
    for n in map(int, counts.split(",")):
        # the long prompt, then the sampled request, then the rest in order
        order = sorted(reqs, key=lambda r: (r["name"] not in ("long_5k", "oracle_long"), r["name"] != "oracle_story_sampled"))
        group = [order[i % len(order)] for i in range(n)]
        for mixed in (False, True) if n > 1 else (False,):
            t0 = time.time()
            got = together(url, group, mixed)
            bad = 0
            for r, ids in zip(group, got):
                if isinstance(ids, Exception) or ids != solo[r["name"]]:
                    bad += 1
                    print(f"   {r['name']}: DIFFERS " + (str(ids) if isinstance(ids, Exception) else sc.first_difference(ids, solo[r["name"]])))
            failed += bad
            print(f"{n} at once{', every second one with draft:false' if mixed else ''}: {n - bad}/{n} equal their solo reply ({time.time() - t0:.1f}s)", flush=True)
    print("PASS" if not failed else f"FAIL: {failed} differences")
    return 1 if failed else 0


USAGE = (
    "Qwen3.6 concurrent streams on the native CUDA server: each stream equals the same request alone.\n"
    "\n"
    "  lanes_check.py URL FIXTURE ORACLE [COUNTS]   COUNTS: stream counts to try, default 1,4,8\n"
    "\n"
    "Requests: the fixture's chat requests (greedy, thinking off) and the oracle's four prompts as token ids, plus\n"
    "one seeded sampled request. Each runs alone first (the reference); then N of them are sent at once, N in COUNTS\n"
    "(the 5k-token prompt among them, so a prefill overlaps the others' decoding). Every reply's ids must equal its\n"
    "solo ids token for token; the oracle prompts' also equal the Python engine's serial reply. Then the same with\n"
    "every second one sent with \"draft\": false. Run the server with --parallel of at least the largest N.\n"
)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(USAGE)
    sys.exit(main(*sys.argv[1:]))
