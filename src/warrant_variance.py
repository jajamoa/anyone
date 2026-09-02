"""Between-user variance in warrant on the SAME post. Zero API cost.

If users who comment on the same post overwhelmingly share a warrant, then no
metric can detect individualization, and the premise of the probe fails.
"""
import json, glob, collections, math, statistics, hashlib

FILES = sorted(glob.glob("ext/suite-colm-data/data/*.json"))
post2 = collections.defaultdict(list)      # scenario-hash -> [(user, warrant, stance)]
warrant_counts = collections.Counter()

for f in FILES:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    uid = d.get("username") or d.get("user_id") or f
    for t in (d.get("topics") or []):
        w = t.get("warrant_gt")
        sc = t.get("scenario_description")
        if not w or not sc:
            continue
        h = hashlib.md5(sc[:400].encode()).hexdigest()[:16]
        post2[h].append((uid, w, t.get("stance_label")))
        warrant_counts[w] += 1

def entropy(xs):
    n = len(xs); c = collections.Counter(xs)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

print(f"labelled comments = {sum(warrant_counts.values())}   posts = {len(post2)}")
print(f"warrant classes = {len(warrant_counts)}   max entropy = {math.log2(len(warrant_counts)):.2f} bits")
print("\nglobal warrant distribution:")
tot = sum(warrant_counts.values())
for w, c in warrant_counts.most_common():
    print(f"   {100*c/tot:5.1f}%  {c:6d}  {w}")

for k in (2, 3, 5):
    grp = [v for v in post2.values() if len({u for u, _, _ in v}) >= k]
    if not grp:
        print(f"\n[!] no post with >={k} distinct users"); continue
    we = [entropy([w for _, w, _ in g]) for g in grp]
    se = [entropy([s for _, _, s in g if s]) for g in grp if any(s for _, _, s in g)]
    uni_w = sum(1 for e in we if e == 0) / len(we)
    uni_s = sum(1 for e in se if e == 0) / len(se) if se else float("nan")
    print(f"\nposts with >={k} distinct users: n={len(grp)}")
    print(f"   WARRANT entropy  mean={statistics.mean(we):.3f} bits   unanimous={100*uni_w:.1f}%")
    print(f"   STANCE  entropy  mean={statistics.mean(se):.3f} bits   unanimous={100*uni_s:.1f}%")
