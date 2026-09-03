"""Step 2. The simulator writes the target's reply under three conditions.

  same   given the target's own comment history
  cross  given another commenter's history on the same post (different person, same post)
  same2  given the target's history again: a fresh sample, the noise floor

Under same and cross the simulator also answers the stance / warrant questions
directly (Method 1). Writes <dir>/generations.json (default data/). Needs data/contexts.json,
which holds the histories and is not distributed.
"""
import json, os, re, sys
from api import call, pmap, report

D = sys.argv[1] if len(sys.argv) > 1 else "data"  # output dir; SUITE_MODEL picks the simulator

SIM = ("You are simulating one specific Reddit user who comments on r/AmItheAsshole. "
       "You are given that person's past comments. Write the comment THIS PERSON would "
       "write for a new scenario: their verdict and their reasoning, in their own voice "
       "and length. Output only the comment text.")
MCQ = ("You are predicting how one specific Reddit user would judge a new r/AmItheAsshole "
       "scenario, based on their past comments. Answer for THAT PERSON, not for yourself "
       "and not for the community consensus.")
TEMPERATURE = 0.1

I = D if os.path.exists(f"{D}/items.json") else "data"  # a data dir may carry its own items / contexts
items = json.load(open(f"{I}/items.json"))
contexts = json.load(open(f"{I}/contexts.json"))


def parse_mcq(text, options):
    m1 = re.search(r"Q1\s*[:.\-]?\s*\**\s*([AB])\b", text, re.I)
    m2 = re.search(r"Q2\s*[:.\-]?\s*\**\s*([ABCD])\b", text, re.I)
    return {"stance": {"A": "NTA", "B": "YTA"}[m1.group(1).upper()] if m1 else None,
            "warrant": options["ABCD".index(m2.group(1).upper())] if m2 else None}


def run(it):
    scenario = it["scenario"][:3500]
    out = {}
    for cond, who in (("same", "same"), ("cross", "cross"), ("same2", "same")):
        history = "Past comments by this person:\n\n" + contexts[it["item_id"]][who]
        text = call(f"New scenario:\n{scenario}\n\nWrite this person's comment.",
                    system=SIM, history=history, max_tokens=400, temperature=TEMPERATURE)
        out[cond] = {"text": text.strip()}
        if cond == "same2":
            continue
        options = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(it["warrant_options"]))
        answer = call(f"New scenario:\n{scenario}\n\n"
                      f"Q1 Stance: what verdict would THIS PERSON give?\nA. NTA\nB. YTA\n\n"
                      f"Q2 Warrant: which principle would THIS PERSON rely on?\n{options}\n\n"
                      f"Answer in the form 'Q1: <letter> Q2: <letter>' and nothing else.",
                      system=MCQ, history=history, max_tokens=16, temperature=TEMPERATURE)
        out[cond]["mcq"] = parse_mcq(answer, it["warrant_options"])
    return it["item_id"], out


if __name__ == "__main__":
    # resumable: finished items are kept in <dir>/generations.partial.json
    part = f"{D}/generations.partial.json"
    done = json.load(open(part)) if os.path.exists(part) else {}
    _lock = __import__("threading").Lock()

    def run_ck(it):
        if it["item_id"] in done:
            return it["item_id"], done[it["item_id"]]
        k, out = run(it)
        with _lock:
            done[k] = out
            json.dump(done, open(part, "w"), indent=1, ensure_ascii=False)
        return k, out
    gens = dict(pmap(run_ck, items))
    json.dump(gens, open(f"{D}/generations.json", "w"), indent=1, ensure_ascii=False)
    report(f"generated {len(gens)} items x 3")
