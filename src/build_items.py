"""Step 1. Build the 60 probe items from the SUITE corpus (private; ask for access).

An item is one (target user, post) pair with two histories:
  same   the target's own past comments, excluding this post
  cross  a matched donor: another corpus user who commented on the same post
         and gave a different warrant

Writes data/items.json (public, target comments included) and data/contexts.json
(the histories; never distributed). Seed fixed; the shipped items.json is the
output of the default arguments.
"""
import json, glob, random, hashlib, collections, argparse

DATA = "ext/suite-colm-data/data/*.json"
WORD_CAP = 6000
MIN_WORDS = 1500
MAX_PER_USER = 4


def post_hash(s):
    return hashlib.md5(s[:400].encode()).hexdigest()[:16]


def load_users():
    users = {}
    for f in sorted(glob.glob(DATA)):
        d = json.load(open(f))
        rows = [t for t in d.get("topics", [])
                if t.get("warrant_gt") and t.get("scenario_description") and t.get("comment_text")]
        if len(rows) >= 8:
            users[d.get("username") or d.get("user_id")] = rows
    return users


def history(rows, exclude):
    out, n = [], 0
    for t in rows:
        if post_hash(t["scenario_description"]) == exclude:
            continue
        block = f"Scenario: {t['scenario_description'][:900]}\nTheir comment: {t['comment_text'][:700]}\n"
        w = len(block.split())
        if n + w > WORD_CAP:
            break
        out.append(block); n += w
    return "\n".join(out), n


def main(n, seed):
    rng = random.Random(seed)
    users = load_users()
    by_post = collections.defaultdict(dict)
    for uid, rows in users.items():
        for t in rows:
            by_post[post_hash(t["scenario_description"])].setdefault(uid, t)

    # One (target, donor) pair per post where the two users' warrants differ.
    cands = []
    for h, d in by_post.items():
        pairs = list(d.items()); rng.shuffle(pairs)
        found = next(((ua, ta, ub, tb) for i, (ua, ta) in enumerate(pairs)
                      for ub, tb in pairs[i + 1:] if ta["warrant_gt"] != tb["warrant_gt"]), None)
        if found:
            cands.append((h, *found))
    rng.shuffle(cands)

    freq = collections.Counter(t["warrant_gt"] for rows in users.values() for t in rows)
    items, contexts, per_user = [], {}, collections.Counter()
    for h, ua, ta, ub, tb in cands:
        if len(items) >= n:
            break
        if per_user[ua] >= MAX_PER_USER:
            continue
        same, sw = history(users[ua], h)
        cross, cw = history(users[ub], h)
        if sw < MIN_WORDS or cw < MIN_WORDS:
            continue
        opts = [ta["warrant_gt"]] + [w for w, _ in freq.most_common() if w != ta["warrant_gt"]][:3]
        rng.shuffle(opts)
        k = f"{h}_{ua}"
        items.append({"item_id": k, "post_hash": h, "target_user": ua, "donor_user": ub,
                      "scenario": ta["scenario_description"], "target_comment": ta["comment_text"],
                      "stance_gt": ta.get("stance_label"), "warrant_gt": ta["warrant_gt"],
                      "donor_warrant": tb["warrant_gt"], "warrant_options": opts,
                      "ctx_same_words": sw, "ctx_cross_words": cw})
        contexts[k] = {"same": same, "cross": cross}
        per_user[ua] += 1

    json.dump(items, open("data/items.json", "w"), indent=1, ensure_ascii=False)
    json.dump(contexts, open("data/contexts.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(items)} items, {len(per_user)} target users, from {len(users)} corpus users")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260902)
    a = ap.parse_args()
    main(a.n, a.seed)
