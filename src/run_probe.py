"""Run the similarity-vs-structure probe under same-person and cross-person context.

Each item is evaluated four ways:
  freeform : generate the response this person would write   -> judged for similarity
  mcq      : pick stance and warrant from fixed options      -> scored for structure
under two conditions (same / cross). A client-side budget guard stops the run
before a spend ceiling is crossed.
"""
import json, os, re, sys, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error

MODEL = "claude-haiku-4-5-20251001"
JUDGE = "claude-haiku-4-5-20251001"
PRICE = {"in": 1.00, "out": 5.00, "cw": 1.25, "cr": 0.10}   # USD per 1M tokens

KEY = re.search(r'SUITE_ANTHROPIC_KEY=(\S+)', open(".env").read()).group(1)
_lock = threading.Lock()
SPEND = {"usd": 0.0, "calls": 0}


def call(model, system_blocks, user, max_tokens, temperature=0.1):
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "system": system_blocks, "messages": [{"role": "user", "content": user}]}
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    last = None
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            u = r["usage"]
            cost = (u["input_tokens"] * PRICE["in"]
                    + u["output_tokens"] * PRICE["out"]
                    + u.get("cache_creation_input_tokens", 0) * PRICE["cw"]
                    + u.get("cache_read_input_tokens", 0) * PRICE["cr"]) / 1e6
            with _lock:
                SPEND["usd"] += cost; SPEND["calls"] += 1
            return r["content"][0]["text"]
        except urllib.error.HTTPError as e:
            last = f"{e.code} {e.read()[:150]}"
            if e.code in (429, 529, 500, 502, 503):
                time.sleep(2 ** attempt + 1); continue
            raise RuntimeError(last)
        except Exception as e:
            last = str(e); time.sleep(2 ** attempt + 1)
    raise RuntimeError(f"exhausted retries: {last}")


GEN_SYS = ("You are simulating one specific Reddit user who comments on r/AmItheAsshole. "
           "You are given that person's past comments. Write the comment THIS PERSON would "
           "write for a new scenario: their verdict and their reasoning, in their own voice "
           "and length. Output only the comment text.")

MCQ_SYS = ("You are predicting how one specific Reddit user would judge a new r/AmItheAsshole "
           "scenario, based on their past comments. Answer for THAT PERSON, not for yourself "
           "and not for the community consensus.")

JUDGE_SYS = ("You compare a predicted comment against the real comment the person actually "
             "wrote. Rate how well the prediction aligns with the real comment in verdict, "
             "reasoning, and voice, on a 1-5 scale where 1 is no alignment and 5 is close "
             "alignment. Output only a single integer.")


def ctx_blocks(ctx):
    return [{"type": "text", "text": GEN_SYS},
            {"type": "text", "text": "Past comments by this person:\n\n" + ctx,
             "cache_control": {"type": "ephemeral"}}]


def run_item(it, cond):
    ctx = it["ctx_same"] if cond == "same" else it["ctx_cross"]
    sc = it["scenario"][:3500]
    rec = {"item_id": it["item_id"], "cond": cond}

    gen = call(MODEL, ctx_blocks(ctx),
               f"New scenario:\n{sc}\n\nWrite this person's comment.", 400)
    rec["gen"] = gen.strip()

    opts = it["warrant_options"]
    letters = "ABCD"
    olines = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
    mcq = call(MODEL,
               [{"type": "text", "text": MCQ_SYS},
                {"type": "text", "text": "Past comments by this person:\n\n" + ctx,
                 "cache_control": {"type": "ephemeral"}}],
               f"New scenario:\n{sc}\n\n"
               f"Q1 Stance: what verdict would THIS PERSON give? A. NTA  B. YTA\n\n"
               f"Q2 Warrant: which principle would THIS PERSON rely on?\n{olines}\n\n"
               f"Answer exactly in the form: Q1: <letter>  Q2: <letter>", 30)
    rec["mcq_raw"] = mcq.strip()
    m1 = re.search(r"Q1\s*[:.\-]?\s*([AB])", mcq, re.I)
    m2 = re.search(r"Q2\s*[:.\-]?\s*([ABCD])", mcq, re.I)
    rec["stance_pred"] = {"A": "NTA", "B": "YTA"}.get(m1.group(1).upper()) if m1 else None
    rec["warrant_pred"] = opts[letters.index(m2.group(1).upper())] if m2 else None
    rec["stance_ok"] = int(rec["stance_pred"] == it["stance_gt"]) if rec["stance_pred"] else 0
    rec["warrant_ok"] = int(rec["warrant_pred"] == it["warrant_gt"]) if rec["warrant_pred"] else 0

    j = call(JUDGE, [{"type": "text", "text": JUDGE_SYS}],
             f"REAL comment:\n{it['target_comment'][:1500]}\n\n"
             f"PREDICTED comment:\n{rec['gen'][:1500]}\n\nScore 1-5:", 8, temperature=0.0)
    jm = re.search(r"[1-5]", j)
    rec["judge"] = int(jm.group()) if jm else None
    return rec


def main(items_path, out_path, budget, workers):
    items = json.load(open(items_path))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line); done.add((r["item_id"], r["cond"]))
            except Exception:
                pass
    jobs = [(it, c) for it in items for c in ("same", "cross")
            if (it["item_id"], c) not in done]
    print(f"{len(items)} items | {len(jobs)} calls-groups pending "
          f"({len(done)} cached) | budget ${budget}")

    fh = open(out_path, "a")
    stop = threading.Event()

    def work(job):
        if stop.is_set():
            return None
        if SPEND["usd"] > budget:
            stop.set(); return None
        try:
            r = run_item(*job)
        except Exception as e:
            return {"item_id": job[0]["item_id"], "cond": job[1], "error": str(e)[:200]}
        return r

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(work, jobs):
            if r is None:
                continue
            fh.write(json.dumps(r) + "\n"); fh.flush(); n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)}  ${SPEND['usd']:.2f}  {SPEND['calls']} calls", flush=True)
    fh.close()
    print(f"\ndone: {n} records | spend ${SPEND['usd']:.3f} over {SPEND['calls']} calls")
    if stop.is_set():
        print("!! stopped early: budget ceiling reached", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/items_pilot.json")
    ap.add_argument("--out", default="results/raw/pilot.jsonl")
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    main(a.items, a.out, a.budget, a.workers)
