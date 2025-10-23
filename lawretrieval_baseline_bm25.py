# For data/lawretrieval_test.py,
# Retrieve 10 documents from data/deduplicated_relevant_laws.json for each query using BM25

import json

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

def main():
    # Load the retrieval base
    with open('data/deduplicated_relevant_laws.json', 'r', encoding='utf-8') as f:
        laws = json.load(f)

    # Initialize the tokenizer and BM25
    tokenizer = Kiwi()
    tokenized_laws = []
    print("BM25 index building...")
    for law in tqdm(laws):
        # print(tokenizer.tokenize(law['text'])); exit()
        meaningful_tokens = [token.form for token in tokenizer.tokenize(law['text']) if token.tag[0] in "NMVX" or token.tag in ["SN", "SL"]]
        tokenized_laws.append(meaningful_tokens)
    bm25 = BM25Okapi(tokenized_laws)
    print("BM25 index built.")

    # Load the test queries
    test_queries = []
    with open('data/lawretrieval_train_groundtruth.jsonl', 'r', encoding='utf-8') as f:
    # with open('data/lawretrieval_test_groundtruth.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            test_queries.append(item)
    
    # Retrieve documents for each query
    results = []
    for query in tqdm(test_queries):
        query_tokens = [token.form for token in tokenizer.tokenize(query["question"]) if token.tag[0] in "NMVX" or token.tag in ["SN", "SL"]]
        top_n_docs = bm25.get_top_n(query_tokens, laws, n=10)
        retrieved_docs = [law["id"] for law in top_n_docs]
        results.append({
            'doc_id': query["doc_id"],
            'question': query["question"],
            'related_laws': retrieved_docs
        })
        print(results[-1])
    
    with open("data/lawretrieval_train_bm25.jsonl", "w", encoding="utf-8") as f:
    # with open("data/lawretrieval_test_bm25.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()