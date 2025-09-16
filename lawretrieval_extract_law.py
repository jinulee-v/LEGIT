import json
import re
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
import pickle

print("Load data...")
with open('data/reasoning_tasks_train.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]
with open('data/reasoning_tasks_test.jsonl', 'r') as file:
    data.extend([json.loads(line) for line in file])

# Collect "relevant_law"
relevant_laws = list()
for datum in data:
    for issue in datum.get('issues', []):
        if 'relevant_law' in issue:
            relevant_laws.extend(issue['relevant_law'])
relevant_laws = set(relevant_laws)  # Remove duplicates
# Filter out very short laws
relevant_laws = [law for law in relevant_laws if len(law) > 20]
# with open("relevant_laws.txt", "w", encoding="utf-8") as f:
#     for law in relevant_laws:
#         f.write(law + "\n")

print("Deduplicate using min hash...")

lsh = MinHashLSH(threshold=0.65, num_perm=64)
minhashes = {}
law_groups = defaultdict(list)
duplicates = set()

def preprocess(text):
    # Remove all punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # remove supreme court citations
    text = re.sub(r'\(대법원.*?참조\)', '', text)
    return text

for law in relevant_laws:
    minhash = MinHash(num_perm=64, seed=42)
    for word in preprocess(law).split():
        minhash.update(word.encode('utf8'))
    try:
        lsh.insert(law, minhash)
    except ValueError as e:
        continue
    minhashes[law] = minhash

print("Cluster...")
# Find duplicates
for law in relevant_laws:
    if law in duplicates:
        continue
    minhash = minhashes[law]
    results = lsh.query(minhash)
    if len(results) > 0:
        # Check if any of results are already in duplicates
        result_duplicates = [r for r in results if r in duplicates]
        if len(result_duplicates) > 0:
            # append law to the existing group
            law_groups[result_duplicates[0]].append(law)
            duplicates.add(law)
        else:
            # Create a new group with the law
            law_groups[law] = results
            # duplicates.add(law)
            duplicates.update(results)

# # Print duplicates
# cnt = 0
# for law, dupes in list(law_groups.items()):
#     if len(dupes) == 2:
#         print(f"0. {dupes[0]}")
#         print(f"1. {dupes[1]}")
#         print("-" * 40)
#         cnt += 1
#         if cnt == 20:
#             break

# Print stats
print("=====")
print(f"Total number of unique relevant laws: {len(relevant_laws)}")
print(f"Total number of law_groups found: {len(law_groups)}")
print(f"Sentences in law_groups: {sum(len(v) for v in law_groups.values())}")
# Size of LSH sets: average, Q0-Q4
lsh_sizes = sorted([len(v) for v in law_groups.values()])
print(f"Average size of LSH sets: {sum(lsh_sizes) / len(law_groups)}")
lsh_sizes_stats = [lsh_sizes[int(len(lsh_sizes) / 4 * i)] for i in range(4)] + [lsh_sizes[-1]] # separate max
print(f"LSH sizes stats: min={lsh_sizes_stats[0]}, Q1={lsh_sizes_stats[1]}, Q2={lsh_sizes_stats[2]}, Q3={lsh_sizes_stats[3]}, max={lsh_sizes_stats[4]}")
# number of groups with more than 2 elements
num_groups_with_more_than_2_elements = sum(1 for v in law_groups.values() if len(v) >= 2)
print(f"Number of groups with more than 2 elements: {num_groups_with_more_than_2_elements}")

# Save deduplicated laws
deduplicated_laws = []
for id, law_group in enumerate(law_groups):
    # Choose the longest law as the representative
    deduplicated_laws.append({
        "id": id, 
        "text": law_group,
        "variants": law_groups[law_group]
    })

with open("data/deduplicated_relevant_laws.json", "w", encoding="utf-8") as f:
    json.dump(deduplicated_laws, f, ensure_ascii=False, indent=4)
# Dump minhash and LSH with pickle
with open("data/lsh_index.pkl", "wb") as f:
    pickle.dump(lsh, f)