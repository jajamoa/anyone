"""Minimal Anthropic Messages client: retries, prompt caching, running cost meter."""
import json, os, re, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

MODEL = os.environ.get("SUITE_MODEL", "claude-haiku-4-5-20251001")
PRICES = {"claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00, "cache_write": 1.25, "cache_read": 0.10},
          "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_write": 3.75, "cache_read": 0.30}}  # USD per 1M tokens
PRICE = PRICES.get(MODEL, PRICES["claude-sonnet-4-6"])

def _key():
    if os.path.exists(".env"):
        m = re.search(r"SUITE_ANTHROPIC_KEY=(\S+)", open(".env").read())
        if m:
            return m.group(1)
    return os.environ["SUITE_ANTHROPIC_KEY"]

_lock = threading.Lock()
spent = {"usd": 0.0, "calls": 0}


def call(user, system=None, history=None, prefill=None, max_tokens=400, temperature=0.0):
    """One completion. `history` is a long text block that gets cached across calls."""
    blocks = [{"type": "text", "text": system}] if system else []
    if history:
        blocks.append({"type": "text", "text": history, "cache_control": {"type": "ephemeral"}})
    messages = [{"role": "user", "content": user}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}
    if blocks:
        body["system"] = blocks
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": _key(), "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=240))
            u = r["usage"]
            with _lock:
                spent["usd"] += (u["input_tokens"] * PRICE["in"] + u["output_tokens"] * PRICE["out"]
                                 + u.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]
                                 + u.get("cache_read_input_tokens", 0) * PRICE["cache_read"]) / 1e6
                spent["calls"] += 1
            return r["content"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529):
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]}") from None
        except Exception:
            pass
        time.sleep(2 ** attempt + 1)
    raise RuntimeError("retries exhausted")


def pmap(fn, xs, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, xs))


def report(stage):
    print(f"{stage}: {spent['calls']} calls, ${spent['usd']:.2f}")


def call_tool(user, system, tool, temperature=0.0, max_tokens=200, cache_prefix=None):
    """One completion forced through a single tool; returns the tool input dict.
    Mirrors SUITE's function-calling answer path (llm_utils.call_with_function)."""
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
            "tools": [tool], "tool_choice": {"type": "tool", "name": tool["name"]}}
    if cache_prefix and user.startswith(cache_prefix):
        # same text, split so the history prefix is cached across the questions that share it
        body["messages"][0]["content"] = [
            {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": user[len(cache_prefix):]}]
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": _key(), "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=240))
            u = r["usage"]
            with _lock:
                spent["usd"] += (u["input_tokens"] * PRICE["in"] + u["output_tokens"] * PRICE["out"]
                                 + u.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]
                                 + u.get("cache_read_input_tokens", 0) * PRICE["cache_read"]) / 1e6
                spent["calls"] += 1
            return next(b["input"] for b in r["content"] if b["type"] == "tool_use")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529):
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]}") from None
        except Exception:
            pass
        time.sleep(2 ** attempt + 1)
    raise RuntimeError("retries exhausted")
