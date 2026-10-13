#!/usr/bin/env python3
"""Qwen3.6 served through the native CUDA server against the Python 0.6.6 engine's recorded answers (stdlib only).

  serve_check.py record URL FIXTURE   against a Python 0.6.6 server (serial, drafts off): writes the fixture
  serve_check.py check URL FIXTURE    against the native server: reruns it and compares
  serve_check.py check URL FIXTURE --draft-off   the same with "draft": false on every request (serial on a drafting server)

The fixture holds each request body, the chat prompt's token ids (/tokenize) and the greedy reply's token ids and
parsed tool calls. `check` compares in this order and exits 1 on any difference: (a) prompt ids, (b) reply ids,
(c) tool calls. Requests marked ids_only (the thinking-on template) skip (b) and (c).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

MAX_TOKENS = 256
OFF = {"enable_thinking": False}

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Current local time in an IANA time zone.",
        "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": ["zone"]},
    },
}


def post(url: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as reply:
        return json.load(reply)


def long_text(url: str, target: int) -> str:
    """Numbered facts, one sentence each, until the chat prompt is about `target` tokens (the fixture keeps the text)."""
    lines: list[str] = []
    while True:
        n = len(lines) + 1
        lines.append(f"Entry {n}: the survey team logged {n * 7 % 97 + 3} samples at station {n * 13 % 41 + 1}, "
                     f"where the reading was {(n * 31 % 1000) / 10:.1f} units and the weather was "
                     f"{['clear', 'overcast', 'windy', 'foggy', 'rainy'][n % 5]}.")
        body = {"messages": [{"role": "user", "content": "\n".join(lines) + "\n\nHow many entries are listed above, and which station is in the last one?"}],
                "chat_template_kwargs": OFF}
        if len(lines) % 20 == 0 and post(url, "/tokenize", body)["count"] >= target:
            return body["messages"][0]["content"]


def requests_for(url: str) -> list[dict]:
    chat = [
        {"name": "system_user", "messages": [
            {"role": "system", "content": "You are a concise assistant. Answer in plain sentences."},
            {"role": "user", "content": "Explain why the sky is blue in three sentences."}]},
        {"name": "multi_turn", "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Mara and I like climbing."},
            {"role": "assistant", "content": "Nice to meet you, Mara. Climbing is a great hobby."},
            {"role": "user", "content": "Suggest two warm-up exercises for me, and use my name."}]},
        {"name": "tools", "tools": [WEATHER, TIME], "messages": [
            {"role": "user", "content": "What is the weather in Paris right now? Use celsius."}]},
        {"name": "tool_result", "tools": [WEATHER, TIME], "messages": [
            {"role": "user", "content": "What is the weather in Paris right now? Use celsius."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 14, \"sky\": \"light rain\"}"},
            {"role": "user", "content": "Thanks. Should I take an umbrella?"}]},
        {"name": "story", "messages": [
            {"role": "user", "content": "Write a detailed story of at least 600 words about a lighthouse keeper who finds a message in a bottle."}]},
        {"name": "long_5k", "messages": [{"role": "user", "content": long_text(url, 5000)}]},
    ]
    out = [dict(r, chat_template_kwargs=OFF) for r in chat]
    out.append({"name": "thinking_on_template", "ids_only": True, "chat_template_kwargs": {"enable_thinking": True},
                "messages": [{"role": "user", "content": "Say hello."}]})
    return out


def tokenize(url: str, r: dict) -> list[int]:
    body = {"messages": r["messages"], "add_generation_prompt": True, "chat_template_kwargs": r["chat_template_kwargs"]}
    if r.get("tools"):
        body["tools"] = r["tools"]
    return post(url, "/tokenize", body)["tokens"]


def complete(url: str, r: dict, draft_off: bool = False) -> dict:
    body = {"messages": r["messages"], "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False,
            "return_token_ids": True, "chat_template_kwargs": r["chat_template_kwargs"]}
    if r.get("tools"):
        body["tools"] = r["tools"]
    if draft_off:
        body["draft"] = False
    t0 = time.time()
    reply = post(url, "/v1/chat/completions", body)
    choice = reply["choices"][0]
    run = reply.get("tensorfold") or {}
    calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
             for c in choice["message"].get("tool_calls") or []]
    return {"ids": run.get("token_ids"), "finish": choice["finish_reason"], "content": choice["message"].get("content"),
            "tool_calls": calls, "tokens_per_second": run.get("tokens_per_second"), "seconds": time.time() - t0}


def record(url: str, path: str) -> int:
    items = []
    for r in requests_for(url):
        r["prompt_ids"] = tokenize(url, r)
        if not r.get("ids_only"):
            answer = complete(url, r)
            r["reply_ids"], r["finish"], r["content"], r["tool_calls"] = answer["ids"], answer["finish"], answer["content"], answer["tool_calls"]
        items.append(r)
        print(f"recorded {r['name']}: {len(r['prompt_ids'])} prompt ids, {len(r.get('reply_ids') or [])} reply ids", flush=True)
    with open(path, "w") as f:
        json.dump({"engine": "python 0.6.6 serial, drafts off", "max_tokens": MAX_TOKENS, "requests": items}, f)
    return 0


def first_difference(a: list[int], b: list[int]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at {i}: {x} vs {y}"
    return f"lengths {len(a)} vs {len(b)}"


def check(url: str, path: str, draft_off: bool = False) -> int:
    fixture = json.load(open(path))
    failed = 0
    stages = {"a": [], "b": [], "c": []}
    for r in fixture["requests"]:
        ids = tokenize(url, r)
        same = ids == r["prompt_ids"]
        stages["a"].append(same)
        print(f"(a) {r['name']}: {'ok' if same else 'DIFF ' + first_difference(ids, r['prompt_ids'])} ({len(ids)} ids)", flush=True)
        if r.get("ids_only"):
            continue
        answer = complete(url, r, draft_off)
        same = answer["ids"] == r["reply_ids"]
        stages["b"].append(same)
        rate = answer["tokens_per_second"]
        print(f"(b) {r['name']}: {'ok' if same else 'DIFF ' + first_difference(answer['ids'] or [], r['reply_ids'])} "
              f"({len(r['reply_ids'])} ids, {rate and round(rate, 1)} tok/s indicative)", flush=True)
        if r.get("tools"):
            calls = answer["tool_calls"] == r["tool_calls"]
            parsed = bool(answer["tool_calls"]) or not r["tool_calls"]
            stages["c"].append(calls and parsed)
            print(f"(c) {r['name']}: {'ok' if calls and parsed else 'DIFF'} native {answer['tool_calls']} python {r['tool_calls']}", flush=True)
    for name, results in stages.items():
        failed += results.count(False)
    print("PASS" if failed == 0 else f"FAIL: {failed} differences")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or sys.argv[1] not in ("record", "check") or (len(sys.argv) == 5 and (sys.argv[1] != "check" or sys.argv[4] != "--draft-off")):
        sys.exit(__doc__)
    if sys.argv[1] == "record":
        sys.exit(record(sys.argv[2], sys.argv[3]))
    sys.exit(check(sys.argv[2], sys.argv[3], len(sys.argv) == 5))
