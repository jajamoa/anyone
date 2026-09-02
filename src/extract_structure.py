"""Method 2: read structure off free text.

For every generated comment (same / cross / same2) and the real comment, a reader
model answers the SUITE stance and warrant MCQ *from the text alone* (no user
history). Structure recovered from text is then scored against the ground truth,
giving a structural fidelity metric that applies to any free-text simulator.
"""
import json, re, sys, time, threading, statistics, collections, random
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error

MODEL = "claude-haiku-4-5-20251001"
PRICE = {"in": 1.00, "out": 5.00}
KEY = re.search(r'SUITE_ANTHROPIC_KEY=(\S+)', open(".env").read()).group(1)
_lock = threading.Lock(); SPEND = {"usd": 0.0, "calls": 0}

WARRANT_DESC = {
    "autonomy_boundaries": "Autonomy/Boundaries: personal sovereignty, the right to decide for oneself, boundaries being violated",
    "property_consent":    "Property/Consent: ownership, money, possessions, use of something without consent",
    "role_obligation":     "Role-based Responsibility: duties that come with a role (parent, partner, host, employee)",
    "care_harm":           "Care/Harm Prevention: concern about emotional or physical suffering",
    "fairness_reciprocity":          "Fairness/Reciprocity: proportionality, equity, reciprocal treatment",
    "tradition_expectations":          "Tradition/Convention: shared social conventions, etiquette, coordination norms",
    "safety_risk":          "Safety/Risk: physical danger, recklessness, risk to wellbeing",
    "honesty_communication":          "Honesty/Trust: truthfulness, deception, broken promises",
    "loyalty_betrayal":          "Loyalty/Ingroup: standing by family, friends, one's group",
    "authority_hierarchy":          "Authority/Respect: deference to legitimate authority or hierarchy",
}
LET = "ABCD"
SYS = ("You are annotating a single Reddit r/AmItheAsshole comment. Read ONLY the comment. "
       "Report the verdict the comment gives and the moral principle the comment itself "
       "relies on to justify that verdict. Do not use your own opinion of the scenario.")


def call(user, prefill, max_tokens=12):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "system": SYS, "messages": [{"role": "user", "content": user},
                                        {"role": "assistant", "content": prefill}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    for a in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120)); u = r["usage"]
            with _lock:
                SPEND["usd"] += (u["input_tokens"]*PRICE["in"] + u["output_tokens"]*PRICE["out"])/1e6
                SPEND["calls"] += 1
            return r["content"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503): time.sleep(2**a+1); continue
            raise RuntimeError(f"{e.code} {e.read()[:150]}")
        except Exception: time.sleep(2**a+1)
    raise RuntimeError("retries exhausted")


def read(it, text):
    opts = it["warrant_options"]
    olines = "\n".join(f"{LET[i]}. {WARRANT_DESC[o]}" for i, o in enumerate(opts))
    out = call(f"Scenario (for reference only):\n{it['scenario'][:2500]}\n\n"
               f"Comment to annotate:\n{text[:2500]}\n\n"
               f"Q1 Verdict given in the comment:\nA. NTA\nB. YTA\n\n"
               f"Q2 Principle the comment relies on:\n{olines}\n\n"
               f"Give only the two letters.", prefill="Q1:")
    txt = "Q1:" + out
    m1 = re.search(r"Q1\s*[:.\-]?\s*\**\s*([AB])\b", txt, re.I)
    m2 = re.search(r"Q2\s*[:.\-]?\s*\**\s*([ABCD])\b", txt, re.I)
    if not m2:
        ls = re.findall(r"\b([ABCD])\b", txt)
        if len(ls) >= 2: m2 = re.match(r"(.)", ls[1])
    st = {"A": "NTA", "B": "YTA"}.get(m1.group(1).upper()) if m1 else None
    wa = opts[LET.index(m2.group(1).upper())] if m2 else None
    return {"raw": txt.strip(), "stance": st, "warrant": wa,
            "stance_ok": int(st == it["stance_gt"]) if st else None,
            "warrant_ok": int(wa == it["warrant_gt"]) if wa else None}


items = {i["item_id"]: i for i in json.load(open("data/items_pilot.json"))}
gens = collections.defaultdict(dict)
for l in open("results/raw/pilot.jsonl"):
    r = json.loads(l); gens[r["item_id"]][r["cond"]] = r["gen"]
for k, g in json.load(open("results/raw/resample.json")).items(): gens[k]["same2"] = g
for k in items: gens[k]["real"] = items[k]["target_comment"]

jobs = [(k, c, t) for k in items for c, t in gens[k].items()]
def do(j):
    k, c, t = j; return (k, c, read(items[k], t))
with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(do, jobs))
out = collections.defaultdict(dict)
for k, c, r in res: out[k][c] = r
json.dump(out, open("results/raw/extracted.json", "w"), indent=1)
print(f"{len(res)} reads | spend ${SPEND['usd']:.3f}\n")

# ---- analysis -----------------------------------------------------------
keys = list(items)
def boot(pairs, B=4000, seed=13):
    rng = random.Random(seed); us = list(pairs)
    obs = statistics.mean([d for u in us for d in pairs[u]]); o = []
    for _ in range(B):
        o.append(statistics.mean([d for _ in us for d in pairs[rng.choice(us)]]))
    o.sort(); return obs, o[int(.025*B)], o[int(.975*B)]
def by_user(f):
    d = collections.defaultdict(list)
    for k in keys: d[items[k]["target_user"]].append(f(k))
    return d

hs = {r["item_id"]: r for r in json.load(open("results/raw/humanlm_scores.json"))}
fails = sum(1 for k in keys for c in out[k] if out[k][c]["warrant_ok"] is None)
print(f"parse failures: {fails}")
print("\nLEVELS (accuracy of structure read off the text, vs GT)")
for q in ("stance_ok", "warrant_ok"):
    row = {c: statistics.mean(out[k][c][q] or 0 for k in keys) for c in ("real", "same", "cross", "same2")}
    print(f"  {q:<11} real={row['real']:.3f}  same={row['same']:.3f}  cross={row['cross']:.3f}  same2={row['same2']:.3f}")
print("\nSAME TEXTS, THREE RULERS:  person effect (same - cross)  vs  noise (same - same2)")
rulers = {
  "HumanLM judge":     lambda k, c: hs[k][c],
  "text -> stance":    lambda k, c: out[k][c]["stance_ok"] or 0,
  "text -> warrant":   lambda k, c: out[k][c]["warrant_ok"] or 0,
}
for name, f in rulers.items():
    pe = boot(by_user(lambda k: f(k, "same") - f(k, "cross")))
    nz = boot(by_user(lambda k: f(k, "same") - f(k, "same2")))
    print(f"  {name:<16} effect {pe[0]:+.3f} [{pe[1]:+.3f},{pe[2]:+.3f}]   noise {nz[0]:+.3f} [{nz[1]:+.3f},{nz[2]:+.3f}]")
# donor shift: does the cross text express the donor's warrant more often?
dw = [k for k in keys if items[k]["donor_warrant"] in items[k]["warrant_options"]]
if dw:
    s = statistics.mean(out[k]["same"]["warrant"] == items[k]["donor_warrant"] for k in dw)
    c = statistics.mean(out[k]["cross"]["warrant"] == items[k]["donor_warrant"] for k in dw)
    print(f"\nDONOR SHIFT (items where donor warrant is an option, n={len(dw)}): "
          f"text expresses donor's warrant  same={s:.3f}  cross={c:.3f}")
