"""Step 4. Read the structured probe off free text (Method 2).

A reader model sees one comment (no history) and answers the same stance /
warrant questions. Runs on the real comment and on each generated text.
Writes data/probe.json.
"""
import json, re, sys
from api import call, pmap, report

WARRANT = {
    "autonomy_boundaries": "Autonomy/Boundaries: personal sovereignty, the right to decide for oneself, boundaries being violated",
    "property_consent": "Property/Consent: ownership, money, possessions, use of something without consent",
    "role_obligation": "Role-based Responsibility: duties that come with a role (parent, partner, host, employee)",
    "care_harm": "Care/Harm Prevention: concern about emotional or physical suffering",
    "fairness_reciprocity": "Fairness/Reciprocity: proportionality, equity, reciprocal treatment",
    "tradition_expectations": "Tradition/Convention: shared social conventions, etiquette, coordination norms",
    "safety_risk": "Safety/Risk: physical danger, recklessness, risk to wellbeing",
    "honesty_communication": "Honesty/Trust: truthfulness, deception, broken promises",
    "loyalty_betrayal": "Loyalty/Ingroup: standing by family, friends, one's group",
    "authority_hierarchy": "Authority/Respect: deference to legitimate authority or hierarchy",
}
READER = ("You are annotating a single Reddit r/AmItheAsshole comment. Read ONLY the comment. "
          "Report the verdict the comment gives and the moral principle the comment itself "
          "relies on to justify that verdict. Do not use your own opinion of the scenario.")

items = json.load(open("data/items.json"))
D = sys.argv[1] if len(sys.argv) > 1 else "data"   # data dir holding generations.json
gens = json.load(open(f"{D}/generations.json"))


def read(it, text):
    options = "\n".join(f"{'ABCD'[i]}. {WARRANT[o]}" for i, o in enumerate(it["warrant_options"]))
    out = "Q1:" + call(f"Scenario (for reference only):\n{it['scenario'][:2500]}\n\n"
                       f"Comment to annotate:\n{text[:2500]}\n\n"
                       f"Q1 Verdict given in the comment:\nA. NTA\nB. YTA\n\n"
                       f"Q2 Principle the comment relies on:\n{options}\n\n"
                       f"Give only the two letters.", system=READER, prefill="Q1:", max_tokens=12)
    m1 = re.search(r"Q1\s*[:.\-]?\s*\**\s*([AB])\b", out, re.I)
    m2 = re.search(r"Q2\s*[:.\-]?\s*\**\s*([ABCD])\b", out, re.I)
    return {"stance": {"A": "NTA", "B": "YTA"}[m1.group(1).upper()] if m1 else None,
            "warrant": it["warrant_options"]["ABCD".index(m2.group(1).upper())] if m2 else None}


def run(it):
    k = it["item_id"]
    texts = {"real": it["target_comment"], **{c: gens[k][c]["text"] for c in ("same", "cross", "same2")}}
    return k, {c: read(it, t) for c, t in texts.items()}


if __name__ == "__main__":
    probe = dict(pmap(run, items))
    json.dump(probe, open(f"{D}/probe.json", "w"), indent=1, ensure_ascii=False)
    report(f"read {len(probe)} items x 4")
