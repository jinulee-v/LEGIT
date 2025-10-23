import json
import os

# Load groundtruth
groundtruth = {} # doc_id -> list of relevant law ids
with open("data/lawretrieval_test_groundtruth.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if len(item["related_laws"]) > 0:
            groundtruth[item["doc_id"]] = item["related_laws"]

# Load retrieval results
results = {} # retriever_name -> doc_id -> list of retrieved law ids
for retriever_name in ["bm25", "contriever", "contriever_finetuned"]:
    results[retriever_name] = {}
    with open(f"data/lawretrieval_test_{retriever_name}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            results[retriever_name][item["doc_id"]] = item["related_laws"]

# Compute recall@10
for retriever_name in ["bm25", "contriever", "contriever_finetuned"]:
    total_recall = 0.0
    for doc_id, relevant_laws in groundtruth.items():
        retrieved_laws = results[retriever_name].get(doc_id, [])
        if not retrieved_laws:
            continue
        hits = sum(1 for law_id in relevant_laws if law_id in retrieved_laws[:10])
        recall = hits / len(relevant_laws)
        total_recall += recall
    avg_recall = total_recall / len(groundtruth)
    print(f"Retriever: {retriever_name}, Recall@10: {avg_recall * 100:.2f}")

# Compute NDCG@10
def dcg(relevances):
    return sum((2**rel - 1) / (2**i) for i, rel in enumerate(relevances, start=1))
def ndcg(retrieved, relevant, k=10):
    relevances = [1 if law_id in relevant else 0 for law_id in retrieved[:k]]
    ideal_relevances = sorted(relevances, reverse=True)
    return dcg(relevances) / dcg(ideal_relevances) if dcg(ideal_relevances) > 0 else 0.0
for retriever_name in ["bm25", "contriever", "contriever_finetuned"]:
    total_ndcg = 0.0
    for doc_id, relevant_laws in groundtruth.items():
        retrieved_laws = results[retriever_name].get(doc_id, [])
        if not retrieved_laws:
            continue
        total_ndcg += ndcg(retrieved_laws, relevant_laws, k=10)
    avg_ndcg = total_ndcg / len(groundtruth)
    print(f"Retriever: {retriever_name}, NDCG@10: {avg_ndcg * 100:.2f}")

# import random
# random.seed(42)
# # Sample 30 examples
# sampled_doc_ids = random.sample(list(groundtruth.keys()), 30)
# # Load ground truth texts
# doc_texts = {} # doc_id -> text
# with open("data/deduplicated_relevant_laws.json", "r", encoding="utf-8") as f:
#     relevant_laws = json.load(f)
#     for law in relevant_laws:
#         doc_texts[law["id"]] = law["text"]
# for retriever_name in ["bm25", "contriever", "contriever_finetuned"]:
#     # Print ground truth laws and retrieved laws for each sample
#     for doc_id in sampled_doc_ids:
#         relevant_laws = groundtruth.get(doc_id, [])
#         retrieved_laws = results[retriever_name].get(doc_id, [])
#         os.system("clear")
#         print(f"Doc ID: {doc_id}, Retriever: {retriever_name}")
#         print(f"  Ground Truth Laws:")
#         for law_id in relevant_laws:
#             print(f"    - {law_id}: {doc_texts.get(law_id, '')}")
#         print(f"  Retrieved Laws:")
#         for law_id in retrieved_laws:
#             print(f"    - {law_id}: {doc_texts.get(law_id, '')}")
#         input("Press Enter to continue...")