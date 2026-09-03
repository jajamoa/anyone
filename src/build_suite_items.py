"""Build free-text items under SUITE's own context construction, for the same users
as the SUITE-protocol MCQ replication (src/suite_mcq.py bench --users N).

  same   SUITE self-warrantaligned/6000: the target's own history, comments whose
         warrant matches this question's answer placed first
  cross  shuffle (default): SUITE shuffle-warrantaligned, a donor from the 694-user pool
         whose dominant warrant differs from the answer, history aligned to that warrant
         derange: another of the run's own users (derangement, seed 13), history aligned
         to the donor's own dominant warrant, test post excluded

Writes <dir>/items.json (public, usernames replaced by user_NNN), <dir>/contexts.json and
<dir>/users.json (never distributed), in the
same schema as data/, so generate.py / judge_humanlm.py / embed.py / probe.py run as is.
"""
import json, random, sys, hashlib
from pathlib import Path
sys.path.insert(0, "src")
import suite_mcq as s

D = sys.argv[1] if len(sys.argv) > 1 else "data/suite_wa"
N_USERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CROSS = sys.argv[3] if len(sys.argv) > 3 else "shuffle"   # shuffle (SUITE's donor) | derange (ours)
ROOT = s.ROOT
KEY_OF = {v.split(":")[0].lower(): k for k, v in s.WARRANT_TEXT.items()}


def opt_key(opt):
    label = opt[3:].split(":")[0].lower()
    return KEY_OF[label]


def blocks(ctx):
    # same layout as build_items.history(): the simulator sees the same pairs SUITE's MCQ saw
    out = "\n".join(f"Scenario: {c['scenario']}\nTheir comment: {c['comment']}\n" for c in ctx)
    return out, len(out.split())


bench = {}
for f in sorted((ROOT / "benchmark/colm-benchmark/self-warrantaligned/6000").glob("*.jsonl")):
    bench[f.stem] = [json.loads(l) for l in open(f)]
users = random.Random(13).sample(sorted(bench), N_USERS)   # same draw as suite_mcq.run_bench
idx = s.donor_index()
if CROSS == "derange":
    dmap, pool = s.derangement(users), s.user_pool(users)

raw = {}
for u in users:
    d = json.load(open(ROOT / "colm-rawdata/colm-modified/user-with-evidence-gt" / f"{u}.json"))
    raw.update({t["comment_id"]: t for t in d["topics"]})

items, contexts = [], {}
for u in users:
    for m in bench[u]:
        if m["question_type"] != "warrant":
            continue
        t = raw[m["comment_id"]]
        wk = s.warrant_key(m)
        if CROSS == "derange":
            donor = dmap[u]
            cross, donor_w = s.derange_context(pool, donor, m["post_id"]), pool[donor]["dominant"]
        else:
            cross, donor = s.shuffle_context(idx, u, m, wk)
            donor_w = next(dw for dw, ds in idx.items() if any(n == donor for n, _, _ in ds))
        same_txt, sw = blocks(m["context"])
        cross_txt, cw = blocks(cross)
        k = f"{m['post_id']}_{u}"
        items.append({"item_id": k, "post_id": m["post_id"], "comment_id": m["comment_id"],
                      "target_user": u, "donor_user": donor,
                      "scenario": m["scenario"], "target_comment": t["comment_text"],
                      "stance_gt": t["stance_label"], "warrant_gt": wk,
                      "donor_warrant": donor_w,
                      "warrant_options": [opt_key(o) for o in m["answer_options"]],
                      "ctx_same_words": sw, "ctx_cross_words": cw})
        contexts[k] = {"same": same_txt, "cross": cross_txt}

# usernames stay private: items.json carries user_NNN, the map lives next to the contexts
names = sorted({i["target_user"] for i in items} | {i["donor_user"] for i in items})
anon = {n: f"user_{k + 1:03d}" for k, n in enumerate(names)}
for i in items:
    old = i["item_id"]
    i["target_user"], i["donor_user"] = anon[i["target_user"]], anon[i["donor_user"]]
    i["item_id"] = f"{i['post_id']}_{i['target_user']}"
    contexts[i["item_id"]] = contexts.pop(old)

Path(D).mkdir(parents=True, exist_ok=True)
json.dump(items, open(f"{D}/items.json", "w"), indent=1, ensure_ascii=False)
json.dump(contexts, open(f"{D}/contexts.json", "w"), indent=1, ensure_ascii=False)
json.dump(anon, open(f"{D}/users.json", "w"), indent=1)
import collections
print(f"{len(items)} items, {len(users)} users; stance {collections.Counter(i['stance_gt'] for i in items)}; "
      f"same words median {sorted(i['ctx_same_words'] for i in items)[len(items)//2]}, "
      f"cross {sorted(i['ctx_cross_words'] for i in items)[len(items)//2]}; "
      f"donor==target warrant: {sum(i['donor_warrant']==i['warrant_gt'] for i in items)}")
