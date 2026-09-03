"""Render docs/items.html: every pilot item with the real comment, the three
generated texts, and their similarity scores. Reuses the stylesheet of
docs/index.html so the two pages match. Reads only distributed files."""
import html, json, re, statistics

items = json.load(open("data/items.json"))
gens = json.load(open("data/generations.json"))
judge = json.load(open("data/judge.json"))
emb = json.load(open("data/embeddings.json"))
probe = json.load(open("data/probe.json"))

css = re.search(r"<style>.*?</style>", open("docs/index.html").read(), re.S).group(0)
EXTRA = """<style>
.item{border:1px solid var(--line);padding:20px 22px;margin-top:22px}
.item h3{font-size:16px;font-weight:700;margin-bottom:4px}
.meta{font-size:12.5px;color:var(--mut);margin-bottom:12px}
.meta span{margin-right:14px}
.gt{border-left:3px solid var(--ink);background:var(--tint);padding:10px 14px;font-size:14px;color:var(--ink);white-space:pre-wrap;margin:10px 0 14px}
.gen{border:1px solid var(--line);padding:10px 14px;margin-top:10px;font-size:13.5px;color:var(--ink2)}
.gen .hd{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:12.5px;margin-bottom:6px}
.gen .hd b{color:var(--ink);font-size:13px}
.gen .sc{color:var(--ink);font-weight:700}
.gen .tx{white-space:pre-wrap}
details{margin-top:8px;font-size:13px;color:var(--ink2)}
summary{cursor:pointer;color:var(--accent-ink);font-size:12.5px}
details .tx{white-space:pre-wrap;margin-top:6px;padding:8px 12px;background:#fafaf8;border:1px solid var(--line)}
.miss{color:#b3261e;font-weight:700}
</style>"""

def esc(s): return html.escape(s or "")
def fmt(x): return f"{x:.2f}"

def mean(cond, src):
    return statistics.mean((src[i["item_id"]][cond]["score"] if src is judge else src[i["item_id"]][cond]) for i in items)

rows, blocks = [], []
for k, it in enumerate(items, 1):
    iid = it["item_id"]; g = gens[iid]; j = judge[iid]; e = emb[iid]; p = probe[iid]
    js = {c: j[c]["score"] for c in ("same", "cross", "same2")}
    rows.append(f"<tr><td><a href='#i{k}'>{k}</a></td><td>{esc(it['target_user'])}</td><td>{esc(it['donor_user'])}</td>"
                f"<td>{fmt(js['same'])}</td><td>{fmt(js['cross'])}</td><td>{fmt(js['same2'])}</td>"
                f"<td>{fmt(e['same'])}</td><td>{fmt(e['cross'])}</td><td>{fmt(e['same2'])}</td>"
                f"<td>{js['same']-js['cross']:+.2f}</td><td>{js['same']-js['same2']:+.2f}</td></tr>")
    gen_html = ""
    for c, label in (("same", "same: the target's own history"),
                     ("cross", f"cross: {esc(it['donor_user'])}'s history, same post"),
                     ("same2", "same2: the target's history, resampled")):
        r = p[c]; st = r["stance"]; wa = r["warrant"]
        stc = "" if st == it["stance_gt"] else " miss"; wac = "" if wa == it["warrant_gt"] else " miss"
        gen_html += (f"<div class='gen'><div class='hd'><b>{label}</b>"
                     f"<span>judge <span class='sc'>{fmt(js[c])}</span> &middot; embedding <span class='sc'>{fmt(e[c])}</span>"
                     f" &middot; read as <span class='{stc.strip()}'>{esc(st)}</span> / <span class='{wac.strip()}'>{esc(wa)}</span></span></div>"
                     f"<div class='tx'>{esc(g[c]['text'])}</div>"
                     f"<details><summary>judge's key points and reasoning</summary><div class='tx'>{esc(j[c]['key_points'])}\n\n{esc(j[c]['thought'])}</div></details></div>")
    blocks.append(f"""<div class="item" id="i{k}">
  <h3>Item {k} &middot; target {esc(it['target_user'])}</h3>
  <div class="meta"><span>stance {esc(it['stance_gt'])}</span><span>warrant {esc(it['warrant_gt'])}</span><span>donor's warrant {esc(it['donor_warrant'])}</span><span>history {it['ctx_same_words']} / {it['ctx_cross_words']} words</span></div>
  <details><summary>the post ({len(it['scenario'].split())} words)</summary><div class="tx">{esc(it['scenario'])}</div></details>
  <div class="gt">{esc(it['target_comment'])}</div>
  {gen_html}
</div>""")

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anyone · all items</title>
<link rel="icon" type="image/svg+xml" href="assets/icon.svg">
{css}
{EXTRA}
</head>
<body>
<div class="topbar"><div class="wrap">
  <b><a href="index.html">Anyone</a></b>
  <nav>
    <a href="index.html#results">Results</a>
    <a href="https://github.com/jajamoa/anyone" target="_blank" rel="noopener">Code</a>
    <a href="https://colab.research.google.com/github/jajamoa/anyone/blob/main/notebooks/anyone.ipynb" target="_blank" rel="noopener">Notebook</a>
  </nav>
  <img class="mitmark" src="assets/mit-black.svg" alt="MIT">
</div></div>

<section class="wrap" style="margin-top:40px">
  <div class="kicker">Appendix</div>
  <h2>All {len(items)} items</h2>
  <p class="body">Raw data from the pilot. For each of the 60 items you can read the post, what the person actually wrote, and what the simulator wrote when given that person's history (same), another commenter's history (cross), or the same history a second time (same2). Next to each reply: the HumanLM judge score, the embedding cosine to the real comment, and the stance and warrant a reader model pulled out of it (red means it differs from the real comment). The histories themselves are not shown.</p>
  <p class="body">Means: judge {fmt(mean('same', judge))} / {fmt(mean('cross', judge))} / {fmt(mean('same2', judge))}, embedding {fmt(mean('same', emb))} / {fmt(mean('cross', emb))} / {fmt(mean('same2', emb))} (same / cross / same2).</p>
  <div class="tablewrap"><table>
    <tr><th>#</th><th>target</th><th>donor</th><th>judge same</th><th>cross</th><th>same2</th><th>emb same</th><th>cross</th><th>same2</th><th>judge same&minus;cross</th><th>same&minus;same2</th></tr>
    {''.join(rows)}
  </table></div>
</section>

<section class="wrap">
{''.join(blocks)}
</section>

<footer><div class="wrap">
  <span>Anyone &middot; 2026</span>
  <span><a href="https://github.com/jajamoa/anyone" target="_blank" rel="noopener">github.com/jajamoa/anyone</a></span>
</div></footer>
</body>
</html>
"""
open("docs/items.html", "w").write(page)
print(f"wrote docs/items.html ({len(page)//1024} KB, {len(items)} items)")
