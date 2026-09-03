"""Step 2. The simulator writes the target's reply under three conditions.

  same   given the target's own comment history
  cross  given another commenter's history on the same post (different person, same post)
  same2  given the target's history again: a fresh sample, the noise floor

Under same and cross the simulator also answers the stance / warrant questions
directly (Method 1). Writes data/generations.json. Needs data/contexts.json,
which holds the histories and is not distributed.
"""
import json, re
from api import call, pmap, report

SIM = ("You are simulating one specific Reddit user who comments on r/AmItheAsshole. "
       "You are given that person's past comments. Write the comment THIS PERSON would "
       "write for a new scenario: their verdict and their reasoning, in their own voice "
       "and length. Output only the comment text.")
MCQ = ("You are predicting how one specific Reddit user would judge a new r/AmItheAsshole "
       "scenario, based on their past comments. Answer for THAT PERSON, not for yourself "
       "and not for the community consensus.")
TEMPERATURE = 0.1

items = json.load(open("data/items.json"))
contexts = json.load(open("data/contexts.json"))


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
                      f"Give only the two letters.",
                      system=MCQ, history=history, prefill="Q1:", max_tokens=12, temperature=TEMPERATURE)
        out[cond]["mcq"] = parse_mcq("Q1:" + answer, it["warrant_options"])
    return it["item_id"], out


if __name__ == "__main__":
    gens = dict(pmap(run, items))
    json.dump(gens, open("data/generations.json", "w"), indent=1, ensure_ascii=False)
    report(f"generated {len(gens)} items x 3")
