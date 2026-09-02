"""Same-person minus cross-person gap, for a similarity metric and structural metrics.

CIs come from a bootstrap clustered on target user, resampling users with
replacement, because items from one person are not independent.
"""
import json, collections, statistics, random, argparse, math

def load(items_path, raw_path):
    items = {i["item_id"]: i for i in json.load(open(items_path))}
    recs = collections.defaultdict(dict)
    err = 0
    for line in open(raw_path):
        r = json.loads(line)
        if "error" in r:
            err += 1; continue
        recs[r["item_id"]][r["cond"]] = r
    paired = {k: v for k, v in recs.items() if "same" in v and "cross" in v}
    return items, paired, err


def boot_ci(by_user, B=4000, seed=7):
    """by_user: {user: [per-item paired differences]} -> (mean, lo, hi)"""
    rng = random.Random(seed)
    users = list(by_user)
    obs = statistics.mean([d for u in users for d in by_user[u]])
    means = []
    for _ in range(B):
        samp = [by_user[rng.choice(users)] for _ in users]
        flat = [d for s in samp for d in s]
        if flat:
            means.append(statistics.mean(flat))
    means.sort()
    return obs, means[int(.025*len(means))], means[int(.975*len(means))]


def report(name, by_user, scale, chance=None):
    obs, lo, hi = boot_ci(by_user)
    flat = [d for v in by_user.values() for d in v]
    sd = statistics.pstdev(flat) or 1e-9
    d = obs / sd
    pct = 100 * obs / scale
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  {name:<26} {obs:+7.3f}  [{lo:+.3f}, {hi:+.3f}]   "
          f"{pct:+6.1f}% of range   d={d:+.2f}{sig}")
    return obs, lo, hi


def main(items_path, raw_path):
    items, paired, err = load(items_path, raw_path)
    print(f"paired items = {len(paired)}   dropped(errors) = {err}")
    users = {items[k]["target_user"] for k in paired}
    print(f"distinct target users = {len(users)}\n")

    lvl = collections.defaultdict(lambda: collections.defaultdict(list))
    diffs = collections.defaultdict(lambda: collections.defaultdict(list))
    nojudge = 0
    for k, v in paired.items():
        u = items[k]["target_user"]
        s, c = v["same"], v["cross"]
        if s.get("judge") is None or c.get("judge") is None:
            nojudge += 1
        else:
            lvl["judge"]["same"].append(s["judge"]); lvl["judge"]["cross"].append(c["judge"])
            diffs["judge"][u].append(s["judge"] - c["judge"])
        for m in ("stance_ok", "warrant_ok"):
            lvl[m]["same"].append(s[m]); lvl[m]["cross"].append(c[m])
            diffs[m][u].append(s[m] - c[m])

    print("LEVELS (same / cross)")
    for m, unit in (("judge", "1-5"), ("stance_ok", "%"), ("warrant_ok", "%")):
        a, b = lvl[m]["same"], lvl[m]["cross"]
        f = (lambda x: 100*statistics.mean(x)) if unit == "%" else (lambda x: statistics.mean(x))
        print(f"  {m:<26} {f(a):6.2f}  /  {f(b):6.2f}   ({unit}, n={len(a)})")
    wg = collections.Counter(items[k]["warrant_gt"] for k in paired)
    print(f"\n  warrant majority-class baseline = {100*max(wg.values())/sum(wg.values()):.1f}%"
          f"   (4 options, chance 25%)")
    if nojudge:
        print(f"  [!] {nojudge} pairs missing a judge score")

    print("\nSAME - CROSS  (paired, bootstrap CI clustered on user)")
    report("similarity (judge 1-5)", diffs["judge"], scale=4.0)
    report("structure: stance", diffs["stance_ok"], scale=1.0)
    report("structure: warrant", diffs["warrant_ok"], scale=1.0)

    su = sum(1 for v in diffs["judge"].values() if statistics.mean(v) > 0)
    wu = sum(1 for v in diffs["warrant_ok"].values() if statistics.mean(v) > 0)
    n = len(diffs["warrant_ok"])
    print(f"\nusers helped by their own context: "
          f"similarity {su}/{len(diffs['judge'])}, warrant {wu}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/items_pilot.json")
    ap.add_argument("--raw", default="results/raw/pilot.jsonl")
    a = ap.parse_args()
    main(a.items, a.raw)
