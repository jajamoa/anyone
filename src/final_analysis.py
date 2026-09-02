"""Can HumanLM's metric detect a person swap?

Three generations per item: same-person, cross-person, and a SECOND same-person
sample. The second same-person sample is the noise floor: whatever separates it
from the first is pure sampling variance, with the person held fixed. A metric
that cannot beat that floor is not measuring the person.
"""
import json, collections, statistics, random, os

items={i["item_id"]:i for i in json.load(open("data/items_pilot.json"))}
R=collections.defaultdict(dict)
for l in open("results/raw/pilot.jsonl"):
    r=json.loads(l); R[r["item_id"]][r["cond"]]=r
resamp=json.load(open("results/raw/resample.json"))
J={s["item_id"]:s for s in json.load(open(os.environ.get("JUDGE","results/raw/humanlm_scores.json")))
   if "parse_error" not in s}
keys=[k for k in J if k in R and "same" in R[k] and "cross" in R[k]]
print(f"n = {len(keys)} items, {len({items[k]['target_user'] for k in keys})} users\n")

# ---------- embedding cosine, as in their Table 3 ----------
from sentence_transformers import SentenceTransformer
import numpy as np
mdl=SentenceTransformer("all-mpnet-base-v2")
texts, idx = [], {}
for k in keys:
    for lab, t in (("gt",items[k]["target_comment"]), ("same",R[k]["same"]["gen"]),
                   ("cross",R[k]["cross"]["gen"]), ("same2",resamp[k])):
        idx[(k,lab)]=len(texts); texts.append(t[:2000])
E=mdl.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
def cos(k,lab): return float(E[idx[(k,lab)]] @ E[idx[(k,"gt")]])

# ---------- assemble ----------
M={"judge (HumanLM rubric)": {l:{k:J[k][l] for k in keys} for l in ("same","cross","same2")},
   "embedding cosine":       {l:{k:cos(k,l) for k in keys} for l in ("same","cross","same2")},
   "structure: warrant":     {l:{k:R[k][c]["warrant_ok"] for k in keys}
                              for l,c in (("same","same"),("cross","cross"))},
   "structure: stance":      {l:{k:R[k][c]["stance_ok"] for k in keys}
                              for l,c in (("same","same"),("cross","cross"))}}

def boot(pairs, B=4000, seed=13):
    """pairs: {user: [diffs]} -> obs, lo, hi"""
    rng=random.Random(seed); us=list(pairs)
    obs=statistics.mean([d for u in us for d in pairs[u]]); out=[]
    for _ in range(B):
        f=[d for _ in us for d in pairs[rng.choice(us)]]
        out.append(statistics.mean(f))
    out.sort(); return obs, out[int(.025*B)], out[int(.975*B)]

def by_user(a,b,f=lambda x,y:x-y):
    d=collections.defaultdict(list)
    for k in keys: d[items[k]["target_user"]].append(f(a[k],b[k]))
    return d

print("LEVELS")
for name,d in M.items():
    s=statistics.mean(d["same"].values()); c=statistics.mean(d["cross"].values())
    extra=f"   same2={statistics.mean(d['same2'].values()):.4f}" if "same2" in d else ""
    print(f"  {name:<24} same={s:.4f}  cross={c:.4f}{extra}")

print("\nPERSON EFFECT (same - cross)   vs   NOISE FLOOR (same - same2)")
print("  a metric that cannot beat its own noise floor is not measuring the person\n")
for name,d in M.items():
    o,lo,hi=boot(by_user(d["same"],d["cross"]))
    sig="" if lo<=0<=hi else "  *"
    line=f"  {name:<24} person {o:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}"
    if "same2" in d:
        o2,lo2,hi2=boot(by_user(d["same"],d["same2"]))
        line+=f"   |   noise {o2:+.4f} [{lo2:+.4f},{hi2:+.4f}]"
    print(line)

print("\nDISCRIMINATION  AUC = P(metric ranks the real person above the alternative)")
print("  0.50 = cannot tell them apart\n")
for name,d in M.items():
    def auc(other):
        pu=by_user(d["same"], d[other], lambda x,y: 1.0 if x>y else (0.0 if x<y else 0.5))
        return boot(pu)
    o,lo,hi=auc("cross")
    line=f"  {name:<24} vs CROSS-PERSON {o:.3f} [{lo:.3f},{hi:.3f}]"
    if "same2" in d:
        o2,lo2,hi2=auc("same2")
        line+=f"   |   vs OWN RESAMPLE {o2:.3f} [{lo2:.3f},{hi2:.3f}]"
    print(line)
