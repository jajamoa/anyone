"""Build probe items: (user, post) pairs with same-person and cross-person context.

Cross-person context is MATCHED: it comes from another real user who commented on
the same post. That is a stronger control than a random stranger, because it rules
out the explanation that a stranger's history is simply irrelevant to the scenario.
"""
import json, glob, random, hashlib, collections, argparse, os

WORD_CAP = 6000
DATA = "ext/suite-colm-data/data/*.json"


def load_users():
    users = {}
    for f in sorted(glob.glob(DATA)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        uid = d.get("username") or d.get("user_id")
        rows = [t for t in (d.get("topics") or [])
                if t.get("warrant_gt") and t.get("scenario_description") and t.get("comment_text")]
        if len(rows) >= 8:
            users[uid] = rows
    return users


def scen_hash(s):
    return hashlib.md5(s[:400].encode()).hexdigest()[:16]


def build_context(rows, exclude_hash, cap=WORD_CAP):
    """Concatenate a user's comments (excluding the target post) up to a word budget."""
    out, n = [], 0
    for t in rows:
        if scen_hash(t["scenario_description"]) == exclude_hash:
            continue
        block = (f"Scenario: {t['scenario_description'][:900]}\n"
                 f"Their comment: {t['comment_text'][:700]}\n")
        w = len(block.split())
        if n + w > cap:
            break
        out.append(block); n += w
    return "\n".join(out), n


def main(n_items, seed, out_path):
    rng = random.Random(seed)
    users = load_users()

    post2users = collections.defaultdict(list)
    for uid, rows in users.items():
        for t in rows:
            post2users[scen_hash(t["scenario_description"])].append((uid, t))

    # Posts commented on by >=2 retained users, where the two differ in warrant.
    # Requiring a warrant difference is what makes the item diagnostic at all:
    # if both users share a warrant, no metric can separate them.
    cands = []
    for h, lst in post2users.items():
        by_user = {}
        for uid, t in lst:
            by_user.setdefault(uid, t)
        if len(by_user) < 2:
            continue
        items = list(by_user.items())
        rng.shuffle(items)
        for i, (ua, ta) in enumerate(items):
            for ub, tb in items[i + 1:]:
                if ta["warrant_gt"] != tb["warrant_gt"]:
                    cands.append((h, ua, ta, ub, tb))
                    break
            else:
                continue
            break
    rng.shuffle(cands)

    warrant_freq = collections.Counter(
        t["warrant_gt"] for rows in users.values() for t in rows)

    items, seen_users = [], collections.Counter()
    for h, ua, ta, ub, tb in cands:
        if len(items) >= n_items:
            break
        if seen_users[ua] >= 4:          # cap items per user so CIs are not driven by one person
            continue
        same_ctx, sw = build_context(users[ua], h)
        cross_ctx, cw = build_context(users[ub], h)
        if sw < 1500 or cw < 1500:
            continue
        gt = ta["warrant_gt"]
        distract = [w for w, _ in warrant_freq.most_common() if w != gt][:3]
        opts = [gt] + distract
        rng.shuffle(opts)
        items.append({
            "item_id": f"{h}_{ua}",
            "post_hash": h,
            "target_user": ua,
            "donor_user": ub,
            "scenario": ta["scenario_description"],
            "target_comment": ta["comment_text"],
            "stance_gt": ta.get("stance_label"),
            "warrant_gt": gt,
            "donor_warrant": tb["warrant_gt"],
            "warrant_options": opts,
            "ctx_same": same_ctx,
            "ctx_cross": cross_ctx,
            "ctx_same_words": sw,
            "ctx_cross_words": cw,
        })
        seen_users[ua] += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(items, open(out_path, "w"), indent=1)
    print(f"retained users        : {len(users)}")
    print(f"candidate posts       : {len(cands)}  (>=2 users, differing warrant)")
    print(f"items written         : {len(items)} -> {out_path}")
    print(f"distinct target users : {len(seen_users)}")
    print(f"ctx words same/cross  : {sum(i['ctx_same_words'] for i in items)/max(1,len(items)):.0f}"
          f" / {sum(i['ctx_cross_words'] for i in items)/max(1,len(items)):.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default="data/items_pilot.json")
    main(*vars(ap.parse_args()).values())
