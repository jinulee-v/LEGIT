"""
Data format:
List of {
    "question": str,
    "positive_ctxs": List[{"title": str, "text": str}],
    "negative_ctxs": List[{"title": str, "text": str}],
    "hard_negative_ctxs": List[{"title": str, "text": str}],
}
"""

import json
import random

with open("data/lawretrieval_train_groundtruth.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f.readlines()]

# with open("data/lawretrieval_train_bm25.jsonl", "r", encoding="utf-8") as f:
#     bm25_data = [json.loads(line) for line in f.readlines()]

with open('data/deduplicated_relevant_laws.json', 'r', encoding='utf-8') as f:
    laws = json.load(f)

with open('data/lawretrieval_train_bm25.jsonl', "r", encoding='utf-8') as f:
    bm25_data = [json.loads(line) for line in f.readlines()]

print(len(data))

processed_data = []
for item, bm25_results in zip(data, bm25_data):
    # print(item.keys()) # ['doc_id', 'question', 'related_laws']
    question = item["question"]
    positive_ctxs = [{"title": "", "text": laws[law_id]['text']} for law_id in item["related_laws"]]
    if len(positive_ctxs) == 0:
        continue

    # Negative contexts:
    # Use BM25 results excluding the positive contexts
    negative_ctxs = []
    for law_id in bm25_results["related_laws"]:
        if law_id not in item["related_laws"]:
            negative_ctxs.append({"title": "", "text": laws[law_id]['text']})

    processed_data.append({
        "question": question,
        "positive_ctxs": positive_ctxs,
        "negative_ctxs": negative_ctxs,
        "hard_negative_ctxs": [],
    })

# Split: train 95%, valid 5%
random.seed(42)
random.shuffle(processed_data)
train_data = processed_data[:int(0.95 * len(processed_data))]
valid_data = processed_data[int(0.95 * len(processed_data)):]

print(f"Train size: {len(train_data)}, Valid size: {len(valid_data)}")

with open("data/contriever_training_data.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open("data/contriever_validation_data.jsonl", "w", encoding="utf-8") as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")