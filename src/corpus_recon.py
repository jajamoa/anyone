"""Zero-cost corpus reconnaissance for the similarity-vs-structure probe."""
import json, glob, collections, statistics, math

FILES = sorted(glob.glob("/Users/jajamoa/MIT/SUITE/colmBackup/*.json"))

users, post2users, n_comments = {}, collections.defaultdict(set), 0
stance_by_post = collections.defaultdict(list)
mcq_tags = collections.Counter()

for f in FILES:
    d = json.load(open(f))
    uid = d["username"]
    topics = d.get("topics") or []
    users[uid] = topics
    n_comments += len(topics)
    for t in topics:
        pid = t.get("post_id")
        if pid:
            post2users[pid].add(uid)
            if t.get("stance_label"):
                stance_by_post[pid].append(t["stance_label"])
        for q in (t.get("mcqs") or []):
            mcq_tags[q.get("tag")] += 1

print(f"users={len(users)}  comments={n_comments}  unique_posts={len(post2users)}")
print(f"comments/user: median={statistics.median(len(v) for v in users.values()):.0f} "
      f"min={min(len(v) for v in users.values())} max={max(len(v) for v in users.values())}")

shared = {p: u for p, u in post2users.items() if len(u) >= 2}
print(f"\nposts with >=2 users in corpus: {len(shared)} "
      f"({100*len(shared)/len(post2users):.1f}% of posts)")
if shared:
    sizes = collections.Counter(len(u) for u in shared.values())
    print("  users-per-shared-post:", dict(sorted(sizes.items())[:8]))

# Between-user disagreement on the SAME post (stance only; warrant needs labels)
def entropy(labels):
    n = len(labels)
    c = collections.Counter(labels)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

ents = [entropy(v) for p, v in stance_by_post.items() if len(v) >= 3]
if ents:
    frac_zero = sum(1 for e in ents if e == 0)/len(ents)
    print(f"\nstance entropy on posts with >=3 users (n={len(ents)}):")
    print(f"  mean={statistics.mean(ents):.3f} bits   fully-unanimous={100*frac_zero:.1f}%")
else:
    print("\n[!] no post has >=3 users: cannot measure between-user stance disagreement")

print("\nmcq tags present:", dict(mcq_tags.most_common(10)))
