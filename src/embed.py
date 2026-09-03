"""Step 5. Embedding cosine between each generated text and the real comment.

Same model HumanLM reports (all-mpnet-base-v2). Writes data/embeddings.json.
"""
import json, sys
from sentence_transformers import SentenceTransformer

items = json.load(open("data/items.json"))
D = sys.argv[1] if len(sys.argv) > 1 else "data"   # data dir holding generations.json
gens = json.load(open(f"{D}/generations.json"))
model = SentenceTransformer("all-mpnet-base-v2")

out = {}
for it in items:
    k = it["item_id"]
    texts = [it["target_comment"]] + [gens[k][c]["text"] for c in ("same", "cross", "same2")]
    e = model.encode([t[:2000] for t in texts], normalize_embeddings=True)
    out[k] = {c: round(float(e[0] @ e[i + 1]), 6) for i, c in enumerate(("same", "cross", "same2"))}
json.dump(out, open(f"{D}/embeddings.json", "w"), indent=1)
print(f"embedded {len(out)} items x 3")
