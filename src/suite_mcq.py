"""SUITE's MCQ protocol, ported verbatim from ext/suite_colm/mcq_evaluator.py.

Prompts, context format, temperature (0.1) and the forced tool-use answer
(`submit_answer` with an enum) are SUITE's; only the transport is ours
(Anthropic Messages instead of the OpenAI-compatible client, which has no
Claude route).

Two sources of questions:
  bench  SUITE's own benchmark dirs, ext/suite_colm/benchmark/colm-benchmark/<strategy>/6000
         conditions: no_context | self-random | self-warrantaligned | shuffle-warrantaligned
                     | derange-warrantaligned
         shuffle = SUITE's warrant-misaligned donor (get_shuffle_context; in practice one or
                   two extreme donors serve everyone)
         derange = ours: a derangement over the run's users, donor history aligned to the
                   donor's own dominant warrant (SUITE's donor construction, ordinary donor)
  items  our 60 items with the same / cross / no_context histories

usage: python src/suite_mcq.py bench <out.json> [--qtypes stance warrant] [--conds ...] [--users N]
       python src/suite_mcq.py items <out.json>
"""
import json, sys, glob, argparse, random, collections
from pathlib import Path
sys.path.insert(0, "src")
import api

ROOT = Path("ext/suite_colm")
TAX = json.load(open(ROOT / "warrant_taxonomy_final.json"))["warrant_taxonomy_v1.2"]["categories"]
WARRANT_TEXT = {k: f"{v['label']}: {v['description']}" for k, v in TAX.items()}
TEMPERATURE = 0.1

BASE_PROMPT = """You are an expert at analyzing personality patterns and predicting individual behavior based on historical data.

TASK OVERVIEW:
You will receive historical comments from a specific person who participates in "Am I The Asshole" (AITA) discussions. Your job is to learn their unique patterns of moral reasoning, values, and judgment style, then predict how they would respond to a NEW scenario they haven't seen before.

CRITICAL UNDERSTANDING:
- This is a PERSONALITY PREDICTION task, not a general moral judgment task
- You must think like the SPECIFIC PERSON whose history you're analyzing
- Different people have different moral frameworks, reasoning styles, and priorities
- Your goal is to capture THEIR unique perspective, not your own or the "correct" answer

ANALYSIS FRAMEWORK:
1. MORAL REASONING PATTERNS: How does this person typically approach ethical dilemmas?
2. VALUE PRIORITIES: What do they consistently care about most? (fairness, autonomy, harm prevention, etc.)
3. EVIDENCE FOCUS: What types of details do they usually emphasize in their reasoning?
4. JUDGMENT STYLE: Are they strict/lenient? Context-sensitive? Consistent across situations?
5. COMMUNICATION PATTERNS: How do they express their views? What language/tone do they use?

"""
SPECIFIC = {
    "stance": """STANCE PREDICTION TASK:
Based on this person's historical patterns, predict whether they would judge the person in the scenario as:
- NTA (Not The Asshole) - The person's actions are justified/acceptable
- YTA (You're The Asshole) - The person's actions are wrong/unacceptable

Focus on how THIS SPECIFIC PERSON typically makes these judgments based on their value system and reasoning patterns.""",
    "warrant": """WARRANT PREDICTION TASK:
Warrants are the underlying moral principles that connect evidence to conclusions. Predict which moral reasoning framework this person would MOST likely use to judge the scenario.

Each option below includes a warrant type and its definition. Analyze which framework best aligns with this person's typical moral reasoning pattern based on their historical comments.""",
}
ROLE_WITH_CTX = """Based on the historical commenting patterns shown above, imagine you are this person. You need to predict how this person would respond to a new scenario they haven't seen before.

First, analyze this person's:
- Typical moral reasoning patterns
- Values and principles they prioritize
- How they evaluate evidence and make judgments
- Their communication style and stance preferences

Then, think step by step:
1. What aspects of the new scenario would this person focus on most?
2. How would this person's values and moral framework apply to this situation?
3. What reasoning process would this person likely use?
4. What conclusion would this person most likely reach?

Now, as this person, respond to the following new scenario:

"""
ROLE_NO_CTX = "Please analyze the following scenario and answer the question:\n\n"
TASK = {"stance": "What stance would this person take on whether the person in the scenario is the asshole?",
        "warrant": "Which moral reasoning warrant would this person most likely use to judge this scenario?"}


def format_context(context):
    if not context:
        return ""
    s = "Historical comments from this person:\n\n"
    for i, item in enumerate(context, 1):
        s += f"Comment {i}:\nScenario: {item['scenario']}\nComment: {item['comment']}\n\n"
    return s


def make_prompt(qtype, scenario, options, context):
    ctx = format_context(context)
    role = ROLE_WITH_CTX if context else ROLE_NO_CTX
    p = f"{ctx}{role}Scenario: {scenario}\n\nQuestion: {TASK[qtype]}\n\nOptions:\n"
    p += "".join(o + "\n" for o in options)
    fmt = "A or B" if qtype == "stance" else "A, B, C, or D"
    p += f"\n\nYou must use the submit_answer tool to provide your answer. Select one letter: {fmt}."
    return p


def tool_for(qtype):
    letters = ["A", "B"] if qtype == "stance" else ["A", "B", "C", "D"]
    return {"name": "submit_answer", "description": "Submit the selected answer option.",
            "input_schema": {"type": "object", "required": ["answer"],
                             "properties": {"answer": {"type": "string", "enum": letters,
                                                       "description": f"The selected answer option: {', '.join(letters)}"}}}}


def ask(qtype, scenario, options, context):
    r = api.call_tool(make_prompt(qtype, scenario, options, context), BASE_PROMPT + SPECIFIC[qtype],
                      tool_for(qtype), temperature=TEMPERATURE, cache_prefix=format_context(context) or None)
    return str(r.get("answer", "")).strip().upper()[:1]


# ---- SUITE's warrant-misaligned donor (get_shuffle_context, cross_context_dir branch) ----
LABEL_TO_KEY = {'property/consent': 'property_consent', 'care/harm': 'care_harm',
                'autonomy/boundaries': 'autonomy_boundaries', 'role-based': 'role_obligation',
                'fairness': 'fairness_reciprocity', 'tradition': 'tradition_expectations',
                'safety': 'safety_risk', 'honesty': 'honesty_communication',
                'relational loyalty': 'loyalty_betrayal', 'authority': 'authority_hierarchy'}


def donor_index():
    idx = collections.defaultdict(list)
    for f in sorted((ROOT / "colm-rawdata/user-with-warrant-gt").glob("*.json")):
        d = json.load(open(f))
        topics = d.get("topics", [])
        cnt = collections.Counter(t.get("warrant_gt") for t in topics if t.get("warrant_gt"))
        if not cnt:
            continue
        dom, n = cnt.most_common(1)[0]
        idx[dom].append((f.stem, n / sum(cnt.values()), topics))
    for k in idx:
        idx[k].sort(key=lambda x: -x[1])
    return dict(idx)


def warrant_key(mcq):
    gt = mcq["answer_options"][ord(mcq["answer"]) - 65].lower()
    return next((k for lab, k in LABEL_TO_KEY.items() if lab in gt), None)


def donor_context(topics, exclude_cid, wkey, budget=6000):
    def w(t): return len(t.get("scenario_description", "").split()) + t.get("comment_length_words", 0)
    valid = [t for t in topics if t.get("comment_id") != exclude_cid
             and t.get("scenario_description") and t.get("comment_text")]
    ctx, used = [], 0
    for pool in ([t for t in valid if t.get("warrant_gt") == wkey], [t for t in valid if t.get("warrant_gt") != wkey]):
        for t in pool:
            wc = w(t)
            if used + wc > budget and used > 0:
                continue
            ctx.append({"scenario": t["scenario_description"], "comment": t["comment_text"]})
            used += wc
            if used >= budget:
                break
    return ctx


def shuffle_context(idx, user, mcq, wk):
    for dom, donors in idx.items():
        if dom == wk:
            continue
        for name, frac, topics in donors:
            if name != user:
                ctx = donor_context(topics, mcq.get("comment_id", ""), dom)
                if ctx:
                    return ctx, name
                break
    return [], None


def derangement(users, seed=13):
    """Donor map over the run's own users: every user gets a different user, nobody gets themselves."""
    rng = random.Random(seed)
    while True:
        perm = users[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(users, perm)):
            return dict(zip(users, perm))


def user_pool(users):
    """Each user's raw history and dominant warrant, from the 694-user pool."""
    pool = {}
    for u in users:
        d = json.load(open(ROOT / "colm-rawdata/user-with-warrant-gt" / f"{u}.json"))
        topics = d.get("topics", [])
        cnt = collections.Counter(t.get("warrant_gt") for t in topics if t.get("warrant_gt"))
        pool[u] = {"topics": topics, "dominant": cnt.most_common(1)[0][0]}
    return pool


def derange_context(pool, donor, post_id):
    """The donor's own history, aligned to the donor's dominant warrant (SUITE's donor
    construction), minus any comment on the test post."""
    topics = [t for t in pool[donor]["topics"] if t.get("post_id") != post_id]
    return donor_context(topics, "", pool[donor]["dominant"])


def run_bench(out, qtypes, conds, nusers):
    def load(strategy):
        d = {}
        for f in sorted((ROOT / "benchmark/colm-benchmark" / strategy / "6000").glob("*.jsonl")):
            d[f.stem] = [json.loads(l) for l in open(f)]
        return d
    bench = {s: load(s) for s in ["self-random", "self-warrantaligned"]}
    users = sorted(bench["self-random"])
    if nusers:
        users = random.Random(13).sample(users, nusers)
    idx = donor_index() if "shuffle-warrantaligned" in conds else None
    if "derange-warrantaligned" in conds:
        dmap, pool = derangement(users), user_pool(users)
    # warrant key per comment_id (SUITE looks it up from the warrant MCQ of the same post)
    wk_of = {m["comment_id"]: warrant_key(m) for u in users for m in bench["self-warrantaligned"][u]
             if m["question_type"] == "warrant"}
    jobs = []
    for cond in conds:
        src = bench["self-warrantaligned" if cond in ("shuffle-warrantaligned", "derange-warrantaligned", "no_context") else cond]
        for u in users:
            for m in src[u]:
                if m["question_type"] not in qtypes:
                    continue
                if cond == "no_context":
                    ctx, donor = [], None
                elif cond == "shuffle-warrantaligned":
                    ctx, donor = shuffle_context(idx, u, m, wk_of.get(m["comment_id"]))
                elif cond == "derange-warrantaligned":
                    donor = dmap[u]
                    ctx = derange_context(pool, donor, m["post_id"])
                else:
                    ctx, donor = m["context"], None
                jobs.append((cond, u, m, ctx, donor))
    print(f"{len(jobs)} calls, {len(users)} users, model {api.MODEL}", flush=True)

    done = checkpoint_load(out)
    jobs = [j for j in jobs if (j[0], j[1], j[2]["comment_id"], j[2]["question_type"]) not in done]
    print(f"{len(done)} done, {len(jobs)} to go", flush=True)

    def do(j):
        cond, u, m, ctx, donor = j
        pred = ask(m["question_type"], m["scenario"], m["answer_options"], ctx)
        r = {"cond": cond, "user": u, "post_id": m["post_id"], "comment_id": m["comment_id"],
             "qtype": m["question_type"], "gt": m["answer"], "pred": pred, "correct": pred == m["answer"],
             "donor": donor, "ctx_items": len(ctx)}
        checkpoint_add(out, r)
        return r
    res = list(done.values()) + api.pmap(do, jobs, workers=6)
    json.dump({"model": api.MODEL, "temperature": TEMPERATURE, "results": res}, open(out, "w"), indent=1)
    summarize(res)
    api.report("bench")


def run_items(out):
    items = json.load(open("data/items.json"))
    ctxs = json.load(open("data/contexts.json"))
    jobs = []
    for it in items:
        opts = [f"{'ABCD'[i]}. {WARRANT_TEXT[o]}" for i, o in enumerate(it["warrant_options"])]
        wgt = "ABCD"[it["warrant_options"].index(it["warrant_gt"])]
        sgt = {"NTA": "A", "YTA": "B"}[it["stance_gt"]]
        for cond in ("same", "cross", "no_context"):
            ctx = [] if cond == "no_context" else parse_history(ctxs[it["item_id"]][cond])
            jobs.append((it, cond, "stance", ["A. NTA (Not The Asshole)", "B. YTA (You're The Asshole)"], sgt, ctx))
            jobs.append((it, cond, "warrant", opts, wgt, ctx))
    print(f"{len(jobs)} calls, model {api.MODEL}", flush=True)

    done = checkpoint_load(out)
    jobs = [j for j in jobs if (j[1], j[0]["target_user"], j[0]["item_id"], j[2]) not in done]
    print(f"{len(done)} done, {len(jobs)} to go", flush=True)

    def do(j):
        it, cond, qt, opts, gt, ctx = j
        pred = ask(qt, it["scenario"], opts, ctx)
        r = {"cond": cond, "user": it["target_user"], "item_id": it["item_id"], "qtype": qt,
             "gt": gt, "pred": pred, "correct": pred == gt, "ctx_items": len(ctx)}
        checkpoint_add(out, r)
        return r
    res = list(done.values()) + api.pmap(do, jobs, workers=6)
    json.dump({"model": api.MODEL, "temperature": TEMPERATURE, "results": res}, open(out, "w"), indent=1)
    summarize(res)
    api.report("items")


_ck = __import__("threading").Lock()


def _ckey(r):
    return (r["cond"], r["user"], r.get("comment_id") or r.get("item_id"), r["qtype"])


def checkpoint_load(out):
    p = Path(out + ".partial.jsonl")
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return {_ckey(r): r for r in rows}


def checkpoint_add(out, r):
    with _ck, open(out + ".partial.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


def parse_history(text):
    """Our contexts.json stores 'Scenario: ...\\nTheir comment: ...' blocks; split back into pairs."""
    out = []
    for block in text.split("Scenario: ")[1:]:
        sc, _, cm = block.partition("\nTheir comment: ")
        out.append({"scenario": sc.strip(), "comment": cm.strip()})
    return out


def summarize(res):
    by = collections.defaultdict(list)
    for r in res:
        by[(r["cond"], r["qtype"])].append(r["correct"])
    for (c, q), v in sorted(by.items()):
        print(f"{c:24s} {q:8s} n={len(v):4d} acc={sum(v)/len(v):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "items"])
    ap.add_argument("out")
    ap.add_argument("--qtypes", nargs="+", default=["stance", "warrant"])
    ap.add_argument("--conds", nargs="+",
                    default=["no_context", "self-random", "self-warrantaligned", "shuffle-warrantaligned"])
    ap.add_argument("--users", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "bench":
        run_bench(a.out, a.qtypes, a.conds, a.users)
    else:
        run_items(a.out)
