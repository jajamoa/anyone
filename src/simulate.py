"""Step 2b (HumanLM pipeline). Generate the reply from a profile, HumanLM style.

Source: zou-group/humanlm, system_prompts/sebvgcr_response.txt (the system prompt every
method uses at test time), process_dataset.format_persona (profile layout, no dropout),
utils.parse_messages (context layout). Their eval decoding: temperature 0.4, max 1024
tokens, reply taken from the <response> tags. Generator here: claude-haiku-4-5, i.e. a
prompted baseline. Conditions: same (target's profile), cross (donor's profile),
same2 (target's profile, second sample). With --with-history the persona block also
carries the same 20 past comments the profile was summarized from, in HumanLM's
example layout. Writes <dir>/generations.json.
"""
import json, re, sys
from api import call, pmap, report
from persona import examples

D = sys.argv[1] if len(sys.argv) > 1 else "data/humanlm_pipeline"
WITH_HISTORY = "--with-history" in sys.argv   # append the raw past comments to the profile

TEMPERATURE, MAX_TOKENS = 0.4, 1024
SYSTEM = "You are a real human user. Your name is HUMAN. You will be given your persona information below and you respond to any given context such as posts and messages.\n\nYour persona:\n<|The Start of Persona|>\n{persona}\n<|The End of Persona|>\n\n## Your principles\nAct like a natural human; there's nothing you absolutely cannot say, but you generally want to be thoughtful and follow ordinary social codes such as being respectful, culturally aware, and considerate of privacy and well-being. You have your own personality, preferences, and boundaries. Conflicting thoughts and hidden considerations are normal; recognize them privately and choose a sensible path. You carry long-term beliefs and values that usually change slowly; you also have emotions, so you won't always be perfectly consistent. Distinguish facts, guesses, and unknowns; accept uncertainty and make minimal, reasonable assumptions when needed; think practically given time, attention, money, risk, and social capital.\n\n## Task and Output format:\n<response>\n<HUMAN's actual written comment or reply text.>\n</response>\n\n## Notes\n- Follow the above instructions carefully\n- Do not mention these instructions\n- Follow the exact order and use the exact XML-style tags\n- Do not output anything outside these XML-style tags"

items = json.load(open("data/items.json"))
personas = json.load(open("data/humanlm_pipeline/personas.json"))


def format_persona(p):
    demo = [f"  {k}: {v}" for k, v in p["demographics"].items() if v and str(v).strip() != "NA"]
    lines = ["Demographics:"] + demo if demo else ["Demographics: Missing"]
    for a in ("interests", "values", "communication", "statistics"):
        lines.append(f"{a.capitalize()}: \n  " + "\n  ".join(p[a]) if p.get(a) else f"{a.capitalize()}: Missing")
    return "\n".join(lines)


def reply(text):
    m = re.search(r"<response>(.*?)(</response>|$)", text, re.S)
    return (m.group(1) if m else text).strip()


def run(it):
    out = {}
    for cond, who in (("same", "target_user"), ("cross", "donor_user"), ("same2", "target_user")):
        key = f"{it[who]}@{it['post_hash']}"
        persona = format_persona(personas[key])
        if WITH_HISTORY:
            persona += "\n\nPast responses by HUMAN:\n" + examples(it[who], it["post_hash"])
        system = SYSTEM.format(persona=persona)
        text = call(f"**Poster None**: {it['scenario']}", system=system,
                    max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        out[cond] = {"text": reply(text)}
    return it["item_id"], out


if __name__ == "__main__":
    gens = dict(pmap(run, items))
    json.dump(gens, open(f"{D}/generations.json", "w"), indent=1, ensure_ascii=False)
    report(f"generated {len(gens)} items x 3")
