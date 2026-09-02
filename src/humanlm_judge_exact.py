"""Score the generations with HumanLM's judge, replicated verbatim.

Source: zou-group/humanlm -> humanlm_train/verl-recipe-humanlm/humanlm/metrics/state_reward.py
(STATE_PROMPT_BATCHED, Apache 2.0) plus train_rl_humanlm.sh (eval_only) and
reward_function.py (n = 1 rollout at validation). Test-time protocol:

  model            anthropic/claude-haiku-4-5, temperature 0, max_tokens 4096
  message          one user turn, no system prompt
  state_name       "response"
  state_desc       "the actual written comment or reply text provided by the user."
  generations      ONE generation per call at test (num_generations = 1), so key
                   points are re-extracted for every call
  context          parse_messages(): "**{Role} None**: {post}" (name is unset upstream)
  other_guidelines the malformed-output / contrast clause from compute_batch_score
  no truncation of context, ground truth, or generation

The earlier src/humanlm_judge.py paraphrased the rubric and batched all three
conditions into one call. This file is the faithful version; both are kept.
"""
import json, re, time, threading, collections, os, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

MODEL = "claude-haiku-4-5-20251001"
PRICE = {"in":1.00,"out":5.00}
KEY = re.search(r'SUITE_ANTHROPIC_KEY=(\S+)', open(".env").read()).group(1)
_lk = threading.Lock(); SPEND={"usd":0.0}
OUT = "results/raw/humanlm_scores_exact.json"

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


def build_prompt(context, ground_truth, generations):
    gd = {i+1: g.strip() for i, g in enumerate(generations)}
    gtxt = (f"<|The Start of Generated {STATE_NAME}s|>\n{json.dumps(gd, indent=2)}\n"
            f"<|The End of Generated {STATE_NAME}s|>")
    return STATE_PROMPT_BATCHED.format(context=context, ground_truth=ground_truth,
        generations_text=gtxt, other_guidelines=OTHER, state_name=STATE_NAME,
        state_desc=STATE_DESC, num_generations=len(generations))


def call(user, max_tokens=4096):
    body={"model":MODEL,"max_tokens":max_tokens,"temperature":0.0,
          "messages":[{"role":"user","content":user}]}
    req=urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key":KEY,"anthropic-version":"2023-06-01","content-type":"application/json"})
    for a in range(5):
        try:
            r=json.load(urllib.request.urlopen(req,timeout=240)); u=r["usage"]
            with _lk: SPEND["usd"]+=(u["input_tokens"]*PRICE["in"]+u["output_tokens"]*PRICE["out"])/1e6
            return r["content"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (429,529,500,502,503): time.sleep(2**a+1); continue
            raise
        except Exception:
            time.sleep(2**a+1)
    raise RuntimeError("retries exhausted")


def extract_json(text):
    m=re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group())


def score_one(context, gt, gen):
    """Test-time protocol: one generation per judge call, up to 5 parse retries."""
    prompt=build_prompt(context, gt, [gen])
    for attempt in range(5):
        out=call(prompt)
        try:
            d=extract_json(out); kp=d.pop("key_points")
            assert len(d)==1 and "score" in d["1"] and "thought" in d["1"]
            return min(max(float(d["1"]["score"]),0.0),1.0), kp, d["1"]["thought"]
        except Exception:
            continue
    return None, None, None


items={i["item_id"]:i for i in json.load(open("data/items_pilot.json"))}
R=collections.defaultdict(dict)
for l in open("results/raw/pilot.jsonl"):
    r=json.loads(l); R[r["item_id"]][r["cond"]]=r
resamp=json.load(open("results/raw/resample.json"))
keys=[k for k,v in R.items() if "same" in v and "cross" in v and k in resamp]

def judge(k):
    it=items[k]
    context=f"**Poster None**: {it['scenario']}"   # parse_messages() layout; author name not kept
    rec={"item_id":k,"info":{}}
    for lab,gen in (("same",R[k]["same"]["gen"]),("cross",R[k]["cross"]["gen"]),("same2",resamp[k])):
        s,kp,th=score_one(context, it["target_comment"], gen)
        if s is None: rec["parse_error"]=lab; continue
        rec[lab]=s; rec["info"][lab]={"key_points":kp,"thought":th}
    return rec

with ThreadPoolExecutor(max_workers=8) as ex:
    scores=list(ex.map(judge, keys))
json.dump(scores, open(OUT,"w"), indent=1)
bad=sum(1 for s in scores if "parse_error" in s)
print(f"judged {len(scores)} items x 3 | parse failures {bad} | ${SPEND['usd']:.3f}")
