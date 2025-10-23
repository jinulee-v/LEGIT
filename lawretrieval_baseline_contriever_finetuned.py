# For data/lawretrieval_test.py,
# Retrieve 10 documents from data/deduplicated_relevant_laws.json for each query using BM25

import argparse
import json
import sys

from tqdm import tqdm, trange
import torch

sys.path.append("contriever")
from contriever.src.contriever import Contriever
from contriever.src.utils import load
from transformers import AutoTokenizer


def main(args):
    print("Load models...")
    device = args.device
    contriever = Contriever.from_pretrained("contriever/checkpoint/my_experiments/checkpoint/step-2000/huggingface").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mcontriever-msmarco")
    # contriever, _, _, _, _ = load(Contriever, "contriever/checkpoint/my_experiments/checkpoint/step-1500", torch.optim.AdamW)

    # Load the retrieval base
    with open('data/deduplicated_relevant_laws.json', 'r', encoding='utf-8') as f:
        laws = json.load(f)
    law_texts = [law['text'] for law in laws]
    law_ids = [law['id'] for law in laws]

    # Encode the law texts
    print("Encoding law texts...")
    law_embeddings = []
    for start in trange(0, len(law_texts), args.encode_batch_size):
        end = min(start + args.encode_batch_size, len(law_texts))
        batch = tokenizer(law_texts[start:end], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            law_embeddings.extend(contriever(**batch))
    law_embeddings = torch.stack(law_embeddings)

    # Load the test queries
    test_queries = []
    with open('data/lawretrieval_test_groundtruth.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            test_queries.append(item)
    
    # Do kNN search
    results = []
    for start in trange(0, len(test_queries), args.knn_batch_size):
        end = min(start + args.knn_batch_size, len(test_queries))
        batch = test_queries[start:end]

        query_tokens = tokenizer([query["question"] for query in batch], return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            query_embedding = contriever(**query_tokens)
        scores = torch.matmul(law_embeddings, query_embedding.T)
        topk_indices = torch.topk(scores, k=10, dim=0).indices
        for i, query in enumerate(batch):
            retrieved_doc_ids = topk_indices[:, i].tolist()
            results.append({
                'doc_id': query["doc_id"],
                'question': query["question"],
                'related_laws': retrieved_doc_ids
            })
    
    # Save results
    with open("data/lawretrieval_test_contriever_finetuned.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for encoding.")
    parser.add_argument("--encode_batch_size", type=int, default=128, help="Batch size for encoding law texts.")
    parser.add_argument("--knn_batch_size", type=int, default=32, help="Batch size for kNN search.")
    args = parser.parse_args()
    main(args)