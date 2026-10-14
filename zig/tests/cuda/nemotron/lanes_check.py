#!/usr/bin/env python3
"""Nemotron concurrent streams on the native CUDA server: two simultaneous greedy replies equal their solo ones."""

from __future__ import annotations

import json
import sys
import threading
import urllib.request

PROMPTS = [
    "Name three colors.",
    "Write one sentence about the sea.",
    "List the first five prime numbers.",
    "Explain in two sentences why the sky is blue.",
    "Say hello in French.",
    "Give a word that rhymes with cat.",
]
MAX_TOKENS = 48


def ask(url: str, prompt: str) -> str:
    """One greedy chat reply's text."""
    messages = [{"role": "user", "content": prompt}]
    body = {"model": "local-model", "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0}
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def pair(i: int) -> tuple[int, int]:
    """Round `i`'s two different prompts."""
    a = i % len(PROMPTS)
    b = (i * 2 + 1) % len(PROMPTS)
    return a, (b + 1) % len(PROMPTS) if a == b else b


def together(url: str, a: int, b: int) -> dict[int, str]:
    """Both prompts sent at the same moment; each one's reply."""
    got: dict[int, str] = {}
    go = threading.Barrier(2)

    def one(i: int) -> None:
        go.wait()
        try:
            got[i] = ask(url, PROMPTS[i])
        except Exception as exc:  # noqa: BLE001
            got[i] = f"error: {exc}"

    threads = [threading.Thread(target=one, args=(i,)) for i in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return got


def main(url: str, pairs: int = 20) -> int:
    solo = [ask(url, p) for p in PROMPTS]
    print(f"-- solo, {len(PROMPTS)} prompts")
    failed = [i for i, text in enumerate(solo) if not text.strip()]
    for i in failed:
        print(f"prompt {i}: empty solo reply")
    bad = 0
    for n in range(pairs):
        a, b = pair(n)
        got = together(url, a, b)
        for i in (a, b):
            same = got[i] == solo[i]
            bad += not same
            if not same:
                print(f"pair {n} prompt {i}: {got[i]!r} != solo {solo[i]!r}")
    total = 2 * pairs
    verdict = "FAIL" if bad or failed else "PASS"
    print(f"{verdict}: {total - bad} of {total} concurrent replies equal the solo reply")
    return 1 if bad or failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: lanes_check.py URL [PAIRS]  (server started with --parallel 2)")
    sys.exit(main(sys.argv[1].rstrip("/"), int(sys.argv[2]) if len(sys.argv) > 2 else 20))
