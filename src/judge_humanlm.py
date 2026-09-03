"""Step 3. Score each generated text with HumanLM's judge, replicated verbatim.

Source: zou-group/humanlm, humanlm/metrics/state_reward.py (STATE_PROMPT_BATCHED,
Apache 2.0), with the test-time settings from train_rl_humanlm.sh and
reward_function.py:

  model        claude-haiku-4-5, temperature 0, max_tokens 4096, no system prompt
  state        name "response", desc "the actual written comment or reply text provided by the user."
  generations  one per call (num_generations = 1 at validation)
  context      parse_messages() layout: "**Poster None**: {post}" (author name unset upstream)
  score        clamp to [0, 1]; up to 5 parse retries

Writes data/judge.json.
"""
import json, re
from api import call, pmap, report

STATE_PROMPT_BATCHED = '''You are a helpful and meticulous evaluator. \
Your task is to score how well the generated {state_name}(s) align with the ground truth user response. \
Description of {state_name}: {state_desc}.

You will be given the context, the ground truth response, and generated {state_name}(s) that you should evaluate.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

{generations_text}

Scoring Criteria:
For each generated {state_name}, assign a score in [0, 1] based on how accurately it reflects the ground truth response.

Guidelines:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response along the {state_name} dimension (e.g., if evaluating a "stance", pick key points related to the stance like "clearly disagrees with X", if evaluating a "response", pick key points about the response like "offers a solution to Y").
   - If {state_name} is different from "a response" (e.g., "stance", "target"), focus on key points only relevant to the {state_name} of the response.
   - Each key point should be specific and distinct.

2. Score how well the generated {state_name} matches each key point:
   - For each key point i, compare it with the generated {state_name} and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K, which measures how comprehensive the generated {state_name} reflects the ground truth response.

4. Compute penalty P for extra or conflicting content:
   - Examine additional content in the generated {state_name} beyond those key points:
     - Does it introduce unsupported evidence and assumptions?
     - Is it irrelevant to what ground truth response expresses?
   - Set a penalty P ∈ [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail; meaning essentially unchanged.
     - [0.4, 0.6]: Moderate speculative or irrelevant content that somewhat shifts emphasis or adds unsupported ideas.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content that clearly changes the meaning.
     - 1.0: Mostly off-topic, contradictory, or dominated by incorrect/hallucinated content.

5. If you are evaluating generated responses (skip if {state_name} is not a response):
   - Length alone does NOT increase the score. Extra length is only ok if it is consistent and not redundant.
   - A generated response that is much longer than the ground truth response should be penalized via P.
   - The generated response may or may not reuse phrases from the context; however, if the generated response just directly copies previous context, without quoting them, treat that as off-task behavior and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))

Additional considerations:
- Follow the instruction carefully.
- Be strict and reserve scores above 0.8 for clearly outstanding matches.
{other_guidelines}

Output format (JSON):
{{
    "key_points": "<analysis of key points from ground truth along {state_name} dimension>",
    "1": {{"thought": "<how well the 1st generated {state_name} matches each key point and compute the final score>", "score": <score>}},
    "2": ...
}}

Format Notes:
- All text in "key_points" and "thought" fields MUST be on a single line with no line breaks or newlines
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (')
- Double check the JSON array's format, especially for the comma and quotation marks
- Ensure that ALL fields, especially "thought" and "score", are present for each item
- You must provide exactly {num_generations} scores for the generated {state_name}(s)

Your output:
'''

STATE_NAME = "response"
STATE_DESC = "the actual written comment or reply text provided by the user."
OTHER = (f"- If a {STATE_NAME} contains non-text content, unnecessary wrappers like XML-like markup, "
         f"or is otherwise malformed, apply a penalty by multiplying its score by 0.5. If there are "
         f"multiple {STATE_NAME}s, you should contrast them against each other to ensure that your "
         f"evaluations are consistent and assign different scores to different generated {STATE_NAME}s.")

items = json.load(open("data/items.json"))
gens = json.load(open("data/generations.json"))


def prompt(context, ground_truth, generation):
    gtxt = (f"<|The Start of Generated {STATE_NAME}s|>\n{json.dumps({1: generation.strip()}, indent=2)}\n"
            f"<|The End of Generated {STATE_NAME}s|>")
    return STATE_PROMPT_BATCHED.format(context=context, ground_truth=ground_truth, generations_text=gtxt,
                                       other_guidelines=OTHER, state_name=STATE_NAME, state_desc=STATE_DESC,
                                       num_generations=1)


def score(context, ground_truth, generation):
    for _ in range(5):
        out = call(prompt(context, ground_truth, generation), max_tokens=4096)
        try:
            d = json.loads(re.search(r"\{.*\}", out, re.S).group())
            return {"score": min(max(float(d["1"]["score"]), 0.0), 1.0),
                    "key_points": d["key_points"], "thought": d["1"]["thought"]}
        except Exception:
            continue
    return None


def run(it):
    k = it["item_id"]
    context = f"**Poster None**: {it['scenario']}"
    return k, {c: score(context, it["target_comment"], gens[k][c]["text"]) for c in ("same", "cross", "same2")}


if __name__ == "__main__":
    judge = dict(pmap(run, items))
    json.dump(judge, open("data/judge.json", "w"), indent=1, ensure_ascii=False)
    report(f"judged {len(judge)} items x 3")
