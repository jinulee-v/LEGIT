import json
import re
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
import pickle
from tqdm import tqdm

print("Load data...")
# Retrieval set extracted from the training set
with open('data/deduplicated_relevant_laws.json', 'r') as file:
    retrieval_base = json.load(file)
    retrieval_base_text_to_id = {item['text']: item['id'] for item in retrieval_base}
# Load LSH index
with open('data/lsh_index.pkl', 'rb') as f:
    lsh: MinHashLSH = pickle.load(f)

def calculate_minhash(text):
    # Preprocessing
    def preprocess(text):
        # Remove all punctuations
        text = re.sub(r'[^\w\s]', '', text)
        # remove supreme court citations
        text = re.sub(r'\(대법원.*?참조\)', '', text)
        return text

    minhash = MinHash(num_perm=64, seed=42)
    for word in preprocess(text).split():
        minhash.update(word.encode('utf8'))
    return minhash
    # usage: lsh.query(calculate_minhash("some text"))


# For training set and test set,
for file in ["data/reasoning_tasks_train.jsonl", "data/reasoning_tasks_test.jsonl"]:
    print(file)
    final_data = []
    total = 0
    missing_data = 0
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    for datum in tqdm(data):
        question = datum['question']
        # Collect "relevant_law"
        relevant_laws = list()
        for issue in datum.get('issues', []):
            if 'relevant_law' in issue:
                relevant_laws.extend(issue['relevant_law'])
        results = set()
        for law in relevant_laws:
            # get the list of top 1 match
            retrieved_results = lsh.query(calculate_minhash(law))
            total += 1
            if len(retrieved_results) <= 0:
                missing_data += 1
            results.update([retrieval_base_text_to_id[r] for r in retrieved_results if r in retrieval_base_text_to_id])
        
        final_data.append({
            "doc_id": datum['doc_id'],
            "question": question,
            "related_laws": list(results)
        })
    with open(file.replace("reasoning_tasks", "lawretrieval"), 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Missing data: {missing_data} out of {total} ({missing_data/total*100:.2f}%)")