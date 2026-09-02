"""Score the existing generations with HumanLM's own rubric (Appendix E.2).

Both conditions are judged inside ONE call, in randomised order, so the judge
extracts key points once and applies them to both. That removes between-call
judge variance, which is the dominant noise source in a paired comparison.
Also adds a same-person resample, to separate the person effect from sampling noise.
"""
import json, re, time, threading, random, urllib.request, urllib.error, collections, os
from concurrent.futures import ThreadPoolExecutor

MODEL = "claude-haiku-4-5-20251001"
PRICE = {"in":1.00,"out":5.00,"cw":1.25,"cr":0.10}
KEY = re.search(r'SUITE_ANTHROPIC_KEY=(\S+)', open(".env").read()).group(1)
_lk = threading.Lock(); SPEND={"usd":0.0}

RUBRIC = """You are given the context, the ground truth response, and generated responses that you should evaluate.

Scoring Criteria:
For each generated response, assign a score in [0, 1] based on how accurately it reflects the ground truth response.

Guidelines:
1. Extract 1-3 key points from the ground truth response. Each key point should be specific and distinct.
2. For each key point i, compare it with the generated response and assign a match value m_i in [0, 1]:
   - 1.0: precisely and perfectly reflected.  [0.7, 0.9]: mostly reflected.
   - [0.4, 0.6]: partially reflected or vague but leaning correct.  [0.1, 0.3]: very weak.
   - 0.0: missed, contradicted, or reversed.
3. Compute coverage C = mean(m_i).
4. Compute penalty P in [0, 1] for extra or conflicting content: unsupported assumptions,
   irrelevance, or content that shifts emphasis. 0.0 none, [0.4,0.6] moderate, 1.0 mostly off-topic.
5. Length alone does NOT increase the score. A generated response much longer than the
   ground truth should be penalized via P.
6. Final score = max(0, min(1, C - P)).
Be strict and reserve scores above 0.8 for clearly outstanding matches.

Output ONLY JSON: {"key_points":"<...>","1":{"thought":"<...>","score":<float>},"2":{"thought":"<...>","score":<float>}}"""


def call(system, user, max_tokens=900):
    body={"model":MODEL,"max_tokens":max_tokens,"temperature":0.0,
          "system":system,"messages":[{"role":"user","content":user}]}
    req=urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key":KEY,"anthropic-version":"2023-06-01","content-type":"application/json"})
    for a in range(5):
        try:
            r=json.load(urllib.request.urlopen(req,timeout=240)); u=r["usage"]
            c=(u["input_tokens"]*PRICE["in"]+u["output_tokens"]*PRICE["out"]
               +u.get("cache_creation_input_tokens",0)*PRICE["cw"]
               +u.get("cache_read_input_tokens",0)*PRICE["cr"])/1e6
            with _lk: SPEND["usd"]+=c
            return r["content"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code in (429,529,500,502,503): time.sleep(2**a+1); continue
            raise
        except Exception:
            time.sleep(2**a+1)
    raise RuntimeError("retries exhausted")


GEN_SYS=("You are simulating one specific Reddit user who comments on r/AmItheAsshole. "
         "You are given that person's past comments. Write the comment THIS PERSON would "
         "write for a new scenario: their verdict and their reasoning, in their own voice "
         "and length. Output only the comment text.")

items={i["item_id"]:i for i in json.load(open("data/items_pilot.json"))}
R=collections.defaultdict(dict)
for l in open("results/raw/pilot.jsonl"):
    r=json.loads(l); R[r["item_id"]][r["cond"]]=r
P={k:v for k,v in R.items() if "same" in v and "cross" in v}
rng=random.Random(5)


def resample(k):
    """Second same-person generation: the sampling-noise control."""
    it=items[k]
    g=call([{"type":"text","text":GEN_SYS},
            {"type":"text","text":"Past comments by this person:\n\n"+it["ctx_same"],
             "cache_control":{"type":"ephemeral"}}],
           f"New scenario:\n{it['scenario'][:3500]}\n\nWrite this person's comment.", 400)
    return k, g.strip()

RS_PATH="results/raw/resample.json"
if os.path.exists(RS_PATH):
    resamp=json.load(open(RS_PATH))
else:
    with ThreadPoolExecutor(max_workers=8) as ex:
        resamp=dict(ex.map(resample, list(P)))
    json.dump(resamp, open(RS_PATH,"w"))
print(f"same-person resamples ready ({len(resamp)}) | ${SPEND['usd']:.3f}")


def judge(k):
    it=items[k]; v=P[k]
    cands=[("same",v["same"]["gen"]), ("cross",v["cross"]["gen"]), ("same2",resamp[k])]
    rng.shuffle(cands)
    gtxt="".join(f"<|The Start of Generated Response {i+1}|>\n{c[1][:1800]}\n"
                 f"<|The End of Generated Response {i+1}|>\n\n" for i,c in enumerate(cands))
    out=call(RUBRIC,
        f"<|The Start of Context|>\n{it['scenario'][:2500]}\n<|The End of Context|>\n\n"
        f"<|The Start of Ground Truth Response|>\n{it['target_comment'][:1500]}\n"
        f"<|The End of Ground Truth Response|>\n\n{gtxt}Your output:")
    m=re.search(r"\{.*\}", out, re.S)
    rec={"item_id":k}
    if m:
        try:
            d=json.loads(m.group())
            for i,(lab,_) in enumerate(cands):
                rec[lab]=float(d[str(i+1)]["score"])
        except Exception as e:
            rec["parse_error"]=str(e)[:120]
    else:
        rec["parse_error"]="no json"
    return rec

with ThreadPoolExecutor(max_workers=8) as ex:
    scores=list(ex.map(judge, list(P)))
json.dump(scores, open("results/raw/humanlm_scores.json","w"), indent=1)
bad=sum(1 for s in scores if "parse_error" in s or len(s)<4)
print(f"judged {len(scores)} items | parse failures {bad} | total ${SPEND['usd']:.3f}")
