"""Compute per-sample STS cosines from test_predictions.jsonl, split by task."""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

PRED_PATH = "/home/wangni/bosch-checkit/results/test_predictions.jsonl"
OUT_PATH = "/home/wangni/notion-figures/bosch-checkit-expert-commentary-generation-for-heat-pump-tim/sts_per_sample.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

records = []
with open(PRED_PATH) as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records)} predictions")

# Determine task from post_prompt
def classify_task(rec):
    pp = rec.get("post_prompt", "").lower()
    if "heizbetrieb" in pp:
        return "space_heating"
    if "warmwasser" in pp:
        return "dhw"
    return "unknown"

gold = [r["gold"].strip() for r in records]
gen = [r["generated"].strip() for r in records]
tasks = [classify_task(r) for r in records]

print("Encoding gold + generated...")
emb_gold = model.encode(gold, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
emb_gen = model.encode(gen, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

cos = (emb_gold * emb_gen).sum(axis=1)  # both normalized → cosine
cos_pct = cos * 100.0

ch = cos_pct[np.array(tasks) == "space_heating"]
dhw = cos_pct[np.array(tasks) == "dhw"]

print(f"CH:  n={len(ch)}  mean={ch.mean():.2f}  std={ch.std(ddof=0):.2f}  min={ch.min():.2f}  max={ch.max():.2f}")
print(f"DHW: n={len(dhw)}  mean={dhw.mean():.2f}  std={dhw.std(ddof=0):.2f}  min={dhw.min():.2f}  max={dhw.max():.2f}")

with open(OUT_PATH, "w") as f:
    json.dump(
        {
            "tasks": tasks,
            "sts_cosine_pct": cos_pct.tolist(),
            "space_heating": ch.tolist(),
            "dhw": dhw.tolist(),
        },
        f,
        indent=2,
    )
print(f"Saved to {OUT_PATH}")
