"""Re-run the MCQ leg with an assistant prefill that forces the answer format.

The first pass let the model open with prose and truncated before any answer,
so 77% of items silently scored zero. Prefill removes the failure mode entirely.
"""
import json, re, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

MODEL = "claude-haiku-4-5-20251001"
PRICE = {"in": 1.00, "out": 5.00, "cw": 1.25, "cr": 0.10}
KEY = re.search(r'SUITE_ANTHROPIC_KEY=(\S+)', open(".env").read()).group(1)
_lock = threading.Lock(); SPEND = {"usd": 0.0}

MCQ_SYS = ("You are predicting how one specific Reddit user would judge a new r/AmItheAsshole "
           "scenario, based on their past comments. Answer for THAT PERSON, not for yourself "
           "and not for the community consensus.")


def call(system, user, prefill, max_tokens=12):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0.1, "system": system,
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": prefill}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    for a in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            u = r["usage"]
            c = (u["input_tokens"]*PRICE["in"] + u["output_tokens"]*PRICE["out"]
                 + u.get("cache_creation_input_tokens",0)*PRICE["cw"]
                 + u.get("cache_read_input_tokens",0)*PRICE["cr"])/1e6
            with _lock: SPEND["usd"] += c
            return r["content"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (429,529,500,502,503): time.sleep(2**a+1); continue
            raise
        except Exception:
            time.sleep(2**a+1)
    raise RuntimeError("retries exhausted")


items = {i["item_id"]: i for i in json.load(open("data/items_pilot.json"))}
recs = [json.loads(l) for l in open("results/raw/pilot.jsonl")]
LET = "ABCD"


def redo(r):
    it = items[r["item_id"]]
    ctx = it["ctx_same"] if r["cond"] == "same" else it["ctx_cross"]
    opts = it["warrant_options"]
    olines = "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(opts))
    out = call([{"type": "text", "text": MCQ_SYS},
                {"type": "text", "text": "Past comments by this person:\n\n" + ctx,
                 "cache_control": {"type": "ephemeral"}}],
               f"New scenario:\n{it['scenario'][:3500]}\n\n"
               f"Q1 Stance: what verdict would THIS PERSON give?\nA. NTA\nB. YTA\n\n"
               f"Q2 Warrant: which principle would THIS PERSON rely on?\n{olines}\n\n"
               f"Give only the two letters.",
               prefill="Q1:")
    txt = "Q1:" + out
    m1 = re.search(r"Q1\s*[:.\-]?\s*\**\s*([AB])\b", txt, re.I)
    m2 = re.search(r"Q2\s*[:.\-]?\s*\**\s*([ABCD])\b", txt, re.I)
    if not m2:                                   # fall back to the 2nd bare letter
        ls = re.findall(r"\b([ABCD])\b", txt)
        if len(ls) >= 2: m2 = re.match(r"(.)", ls[1])
    r["mcq_raw"] = txt.strip()
    r["stance_pred"] = {"A": "NTA", "B": "YTA"}.get(m1.group(1).upper()) if m1 else None
    r["warrant_pred"] = opts[LET.index(m2.group(1).upper())] if m2 else None
    r["stance_ok"] = int(r["stance_pred"] == it["stance_gt"]) if r["stance_pred"] else None
    r["warrant_ok"] = int(r["warrant_pred"] == it["warrant_gt"]) if r["warrant_pred"] else None
    return r


with ThreadPoolExecutor(max_workers=8) as ex:
    recs = list(ex.map(redo, recs))

fails = sum(1 for r in recs if r["stance_pred"] is None or r["warrant_pred"] is None)
with open("results/raw/pilot.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
print(f"re-ran {len(recs)} MCQs | parse failures now {fails} | spend ${SPEND['usd']:.3f}")
