"""Step 2a (HumanLM pipeline). Summarize each history into a user profile, HumanLM style.

Source: zou-group/humanlm, humanual_datasets/persona_generator.py (PERSONA_PROMPT_TEMPLATE)
and process_raw.py (example layout). Their settings: claude-haiku-4-5, temperature 0,
max_tokens 4096, the user's earliest 20 responses, each truncated to 1024 words.
The SUITE corpus has no timestamps, so we take the first 20 rows in corpus order and
omit the Timestamp line. Writes data/humanlm_pipeline/personas.json (not distributed:
the profiles quote the histories).
"""
import json, re, glob
from api import call, pmap, report
from build_items import DATA, post_hash

APP = "Humanual-Opinion"
N_HISTORY, WORD_CAP = 20, 1024

PERSONA_PROMPT_TEMPLATE = """You are an expert at analyzing a {app_name} user behavior. You should generate a JSON object to describe user persona based a target user's responses to some contexts. The contexts ONLY provide other people' posts, and you should NOT use them to infer the target user's demographics. You should ONLY use the target user's responses to summarize the persona.

## Context and Responses:
{comments_text}

## Aspects to cover:

1. Demographics:
- Use explicit subfields: "age group", "gender", "location", "occupation", "nationality", "other"
- Fill with explicit info if available, otherwise "NA".

2. Interests:
- What subjects or themes do they frequently respond on?

3. Values:
- What opinions, attitudes, or worldviews are reflected in their responses?

4. Communication:
- What are their writing styles and formatting habits?

5. Statistics:
- Average / Minimum / Maximum response length (in words). Most frequent words or phrases. Variations in sentence structure and so on.

## Output (strict JSON):
{{
    "analysis": <str>,
    "demographics": {{
        "age group": <str>,
        "gender": <str>,
        "location": <str>,
        "occupation": <str>,
        "nationality": <str>,
        "other": <str>
    }},
    "interests": <a list of 8-12 phrases>,
    "values": <a list of 8-12 phrases>,
    "communication": <a list of 8-12 phrases>,
    "statistics": <a list of 5-10 phrases>
}}

## Instructions:
- [CRITICAL] You MUST always include ALL fields in the JSON output, including "demographics" with ALL its subfields. If demographic information is not explicitly mentioned in the user's responses, set all demographic fields to "NA" but still include them.
- "age group" field: Identify if the user mentioned being X years old in a response from year Y. And find the year of their last response, say Z. Then calculate their age group as (X + (Z - Y)). If no explicit age mentioned, set to "NA".
- "demographics" fields: When extracting demographics, only use explicitly mentioned information. Base your evidence on the user's responses. Do not make assumptions or guesses. If no explicit information is available, use "NA" for each field but ALWAYS include the demographics object.
- [Important!] Other fields: Ensure the phrases are specific, evidence-based, and describe comprehensive aspects of the user. You should quote parts of the user's actual responses as evidence in each phrase without metionining the example index. Avoid vague or generic phrases. Instead, reflect the user's unique traits, behaviors, or preferences.
- "analysis" field: Provide a detailed and step-by-step analysis with the evidence and your reasoning to obtain the user's demongraphics, interests, values, communication style, and statistics.

Your Output:
"""

items = json.load(open("data/items.json"))
users = {}
for f in sorted(glob.glob(DATA)):
    d = json.load(open(f))
    users[d.get("username") or d.get("user_id")] = d.get("topics", [])


def examples(uid, exclude):
    rows = [t for t in users[uid] if t.get("comment_text") and post_hash(t["scenario_description"]) != exclude][:N_HISTORY]
    return "".join(f"<|Start of Example|>\n<|Start of the Context|>\n**Poster None**: {t['scenario_description']}\n"
                   f"<|End of the Context|>\n\n<|Start of the Target User's Response|>\n"
                   f"{' '.join(t['comment_text'].split()[:WORD_CAP])}\n<|End of the Target User's Response|>\n\n"
                   f"<|End of Example|>\n\n" for t in rows)


def persona(key):
    uid, exclude = key.rsplit("@", 1)
    prompt = PERSONA_PROMPT_TEMPLATE.format(app_name=APP, comments_text=examples(uid, exclude))
    for _ in range(5):
        try:
            p = json.loads(re.search(r"\{.*\}", call(prompt, max_tokens=4096), re.S).group())
            p.pop("analysis", None)
            assert set(p) == {"demographics", "interests", "values", "communication", "statistics"}
            return key, p
        except Exception:
            continue
    raise RuntimeError(f"persona failed for {key}")


if __name__ == "__main__":
    keys = sorted({f"{it[w]}@{it['post_hash']}" for it in items for w in ("target_user", "donor_user")})
    out = dict(pmap(persona, keys))
    json.dump(out, open("data/humanlm_pipeline/personas.json", "w"), indent=1, ensure_ascii=False)
    report(f"{len(out)} personas")
