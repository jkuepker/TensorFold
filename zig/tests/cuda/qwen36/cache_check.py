#!/usr/bin/env python3
"""Qwen3.6 prompt cache on the native CUDA server: the next turn with the cache on equals the cache off (stdlib)."""

from __future__ import annotations

import json
import sys
import urllib.request

from serve_check import OFF, long_text, post

MAX_TOKENS = 128
TARGET = 5000

CHATS = {
    "short": [
        ("You are a concise assistant. Answer in plain sentences.", "Name three rivers in Europe and say which is the longest."),
        ("The Volga is the longest of them: about 3,530 km, against roughly 2,850 km for the Danube and 1,230 km for the Rhine.", "Which of those flows through the most countries?"),
    ],
    "three_turns": [
        ("You are a helpful assistant that keeps answers short.", "I am planning a small vegetable garden. Which three vegetables are easiest to start with?"),
        ("Lettuce, radishes and green beans are forgiving: they sprout fast, need little space and tolerate uneven watering.", "How far apart should I plant the radishes?"),
        ("Sow radish seeds about 2 cm deep and thin the seedlings to roughly 5 cm apart, in rows 15 cm apart.", "And how long until I can harvest them?"),
    ],
}


def turns(history: list[tuple[str, str]]) -> list[list[dict]]:
    """Message lists, one per turn: the conversation's system text, user questions and the assistant texts between."""
    system, first = history[0]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    out = [list(messages)]
    for answer, question in history[1:]:
        messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": question}]
        out.append(list(messages))
    return out


def conversations(url: str) -> dict[str, list[list[dict]]]:
    out = {name: turns(history) for name, history in CHATS.items()}
    text = long_text(url, TARGET)
    long_history = [("You answer questions about survey logs.", text),
                    ("There are many entries, each listing a station, a reading and the weather.", "Which weather did the last entry log?")]
    out["long"] = turns(long_history)
    return out


def complete(url: str, messages: list[dict]) -> dict:
    body = {"messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False,
            "return_token_ids": True, "chat_template_kwargs": OFF}
    reply = post(url, "/v1/chat/completions", body)
    usage = reply["usage"]
    return {"prompt": usage["prompt_tokens"], "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "ids": reply["tensorfold"]["token_ids"], "seconds": reply["tensorfold"].get("ttft")}


def metric(url: str, name: str) -> float | None:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=60) as reply:
        for line in reply.read().decode().splitlines():
            if line.split(" ")[0].endswith(name):
                return float(line.split(" ")[1])
    return None


def run(url: str) -> list[dict]:
    out = []
    for name, chat in conversations(url).items():
        for turn, messages in enumerate(chat):
            answer = complete(url, messages)
            answer.update(conversation=name, turn=turn, messages=messages)
            out.append(answer)
            print(f"{name} turn {turn}: {answer['prompt']} prompt tokens, {answer['cached']} cached, {len(answer['ids'])} reply ids", flush=True)
    return out


def record(url: str, path: str) -> int:
    turns_ = run(url)
    if any(t["cached"] for t in turns_):
        sys.exit("record against a server run with --prompt-cache-gib 0: a turn restored kept tokens")
    with open(path, "w") as f:
        json.dump({"max_tokens": MAX_TOKENS, "turns": turns_}, f)
    return 0


def first_difference(a: list[int], b: list[int]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at {i}: {x} vs {y}"
    return f"lengths {len(a)} vs {len(b)}"


def check(url: str, path: str) -> int:
    fixture = json.load(open(path))
    before = metric(url, "prefix_cache_hits_total")
    failed = 0
    cached = 0
    for want in fixture["turns"]:
        got = complete(url, want["messages"])
        same = got["ids"] == want["ids"] and got["prompt"] == want["prompt"]
        hit = want["turn"] == 0 or got["cached"] > 0
        cached += got["cached"]
        print(f"{want['conversation']} turn {want['turn']}: {'ok' if same else 'DIFF ' + first_difference(got['ids'], want['ids'])} "
              f"({got['prompt']} prompt tokens, {got['cached']} cached{'' if hit else ', NO HIT'}, {len(got['ids'])} reply ids)", flush=True)
        failed += (not same) + (not hit)
    after = metric(url, "prefix_cache_hits_total")
    counted = None if before is None or after is None else after - before
    print(f"/metrics prefix_cache_hits_total rose by {counted}; the turns' cached_tokens sum {cached}")
    if counted != cached or cached == 0:
        failed += 1
    print("PASS" if failed == 0 else f"FAIL: {failed} problems")
    return 1 if failed else 0


USAGE = (
    "Qwen3.6 prompt cache on the native CUDA server: the next turn with the cache on equals the cache off.\n"
    "\n"
    "  cache_check.py record URL FIXTURE   against a server run with --prompt-cache-gib 0: writes each turn's ids\n"
    "  cache_check.py check URL FIXTURE    against a server with the cache on: reruns the turns and compares\n"
    "\n"
    "Three conversations (a short chat, a three-turn chat, a ~5k-token history) are sent turn by turn, each turn's\n"
    "history the fixed assistant text below, so both servers see the same prompts. `check` exits 1 unless\n"
    "every turn's reply ids equal the recorded ones token for token, every turn after a conversation's first\n"
    "restored kept tokens (usage's cached_tokens), and /metrics counts exactly those tokens as prefix cache hits.\n"
)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("record", "check"):
        sys.exit(USAGE)
    sys.exit((record if sys.argv[1] == "record" else check)(sys.argv[2], sys.argv[3]))
