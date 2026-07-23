#!/usr/bin/env python3
"""Build an expert topic-affinity atlas for the running pulsar model.

No engine changes: route each topic's prompts through /v1/chat/completions,
diff the /experts routing heat before/after to get that topic's per-expert
routing, assemble a [cell x topic] affinity matrix, then PCA to 2D. VRAM-
resident generalists show ~no per-topic heat signal (served from the tier,
not the host cache) so they collapse to the core; disk/RAM specialists carry
the topic signal and spread out - exactly the colibri galaxy structure.
"""
import json, sys, time, urllib.request
import numpy as np

BASE = "http://100.84.87.107:11435"
OUT = "/mnt/models/atlas.json"

TOPICS = {
    "poetry": ["Write a short poem about the sea at dawn.", "Compose four lines of verse about autumn leaves.", "Write a haiku about a mountain river.", "Write a rhyming couplet about the moon."],
    "law": ["Explain the concept of consideration in contract law.", "What is the difference between negligence and strict liability?", "Summarize the elements of a valid contract.", "Explain what 'mens rea' means in criminal law."],
    "code_python": ["Write a Python function to reverse a linked list.", "Write Python code to read a CSV and compute column means.", "Implement binary search in Python.", "Write a Python decorator that memoizes a function."],
    "math_proof": ["Prove that the square root of 2 is irrational.", "Prove that there are infinitely many prime numbers.", "Show that the sum of the first n integers is n(n+1)/2.", "Prove that a continuous function on a closed interval is bounded."],
    "chinese": ["用中文写一段关于长城的介绍。", "请用中文解释什么是人工智能。", "用中文写一首关于春天的短诗。", "用中文描述一下北京的秋天。"],
    "medicine": ["Explain the mechanism of action of beta blockers.", "What are the stages of wound healing?", "Describe how insulin regulates blood glucose.", "Explain the pathophysiology of hypertension."],
    "json_format": ["Return a JSON object describing a book with title, author, and year.", "Output a JSON array of three users each with name and email.", "Give a JSON schema for a product with price and tags.", "Return JSON for a weather report with temperature and condition."],
    "casual_chat": ["Hey, how's your day going?", "What did you have for breakfast?", "Any fun weekend plans?", "Tell me something interesting you learned recently."],
    "german": ["Schreibe einen kurzen Absatz über München.", "Erkläre auf Deutsch, was maschinelles Lernen ist.", "Schreibe ein kurzes Gedicht über den Wald.", "Beschreibe das Wetter im deutschen Herbst."],
    "sql": ["Write a SQL query to find the top 5 customers by total order value.", "Write SQL to join orders and customers and filter by date.", "Write a SQL query with a GROUP BY and HAVING clause.", "Write SQL to create an index on a users email column."],
}


def get_experts():
    with urllib.request.urlopen(BASE + "/experts", timeout=20) as r:
        d = json.load(r)
    heat, tier = [], []
    for L in d["layers"]:
        heat += L["heat"]
        tier += L["tier"]
    return np.array(heat, dtype=np.float64), np.array(tier, dtype=np.int32)


def chat(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        r.read()


topics = list(TOPICS)
n_topic = len(topics)
aff = None
tier = None
for ti, topic in enumerate(topics):
    before, tier = get_experts()
    if aff is None:
        aff = np.zeros((before.shape[0], n_topic), dtype=np.float64)
    for p in TOPICS[topic]:
        try:
            chat(p)
        except Exception as e:
            print(f"  chat fail ({topic}): {e}", flush=True)
    after, _ = get_experts()
    aff[:, ti] = np.maximum(0.0, after - before)
    print(f"[{ti+1}/{n_topic}] {topic}: routed cells {(aff[:,ti]>0).sum()}", flush=True)

# per-expert stats
total = aff.sum(axis=1)
routed = total > 0
row_norm = np.where(total[:, None] > 0, aff / np.maximum(total[:, None], 1e-9), 0.0)
dom = aff.argmax(axis=1).astype(int)
mx = aff.max(axis=1)
# specialization: concentration above uniform, 0 (generalist) .. 1 (pure)
conc = np.where(total > 0, mx / np.maximum(total, 1e-9), 0.0)
spec = np.clip((conc - 1.0 / n_topic) / (1.0 - 1.0 / n_topic), 0.0, 1.0)

# PCA of the normalized affinity to 2D
X = row_norm - row_norm.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
coords = U[:, :2] * S[:2]  # project
# generalists (no signal) → jitter near origin so they read as a faint core cloud
rng = np.random.default_rng(0)
jit = rng.normal(0, 0.02, size=coords.shape)
coords = np.where(routed[:, None], coords, jit)
# normalize to ~[-1,1]
mabs = np.abs(coords).max() or 1.0
coords = coords / mabs

# topic anchors: PCA projection of each topic's one-hot (unit vector), same transform
onehot = np.eye(n_topic) - row_norm.mean(axis=0, keepdims=True)
anchor_xy = (onehot @ Vt[:2].T) / mabs

experts = [
    {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4),
     "t": int(dom[i]), "spec": round(float(spec[i]), 3), "heat": int(total[i])}
    for i in range(coords.shape[0])
]
n_spec = int((spec >= 0.5).sum())
atlas = {
    "topics": topics,
    "experts": experts,
    "anchors": [{"label": topics[k], "x": round(float(anchor_xy[k, 0]), 4), "y": round(float(anchor_xy[k, 1]), 4)} for k in range(n_topic)],
    "n_experts": len(experts),
    "n_specialists": n_spec,
}
with open(OUT, "w") as f:
    json.dump(atlas, f)
print(f"ATLAS-DONE {len(experts)} experts, {n_spec} specialists -> {OUT}", flush=True)
