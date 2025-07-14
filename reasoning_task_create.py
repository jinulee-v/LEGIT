import os
from tqdm import tqdm
import json
import random
random.seed(42)

def main():
    # Load the dataset
    data_corpus_raw = []
    with open("data/legit.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            datum = json.loads(line)
            data_corpus_raw.append(datum)

     # for testing
    # data_corpus = data_corpus_raw[:9]
    # data_corpus += [x for x in data_corpus_raw if x["doc_id"] == "수원지방법원-2021구단597"]  # Duplicate a specific datum for testing
    data_corpus = data_corpus_raw

    # Extract issues
    tasks = []
    for datum in data_corpus:
        root_issue = None
        for issue in datum["issues"]:
            issue_id = issue["id"]
            if issue_id == "":
                root_issue = issue
                break
        if root_issue is None:
            print(f"Warning: No root issue found in datum {datum['doc_id']}. Skipping...")
            continue
        
        question = "[사실관계]\n" + datum["event_summary"] + "\n\n[청구취지]\n"
        for claim in root_issue["claim"]:
            question += f"- {claim['claimer']}: {claim['content']}\n"
        question += "\n위 재판의 결과를 예측하시오."
        question = question.strip()
        task = {
            "doc_id": datum["doc_id"],
            "issue_id": issue_id,
            "question": question,
            "answer": root_issue["conclusion"],
            "issues": datum["issues"]
        }
        tasks.append(task)

    # Sample the test set: 100 from issue count <= 4, 100 from 4<issue count <= 8, and 100 from issue count > 8
    tasks_issue_count = {"easy": [], "medium": [], "hard": []}
    for task in tasks:
        issue_count = len(task["issues"])
        if issue_count <= 4:
            task["difficulty"] = "easy"
            tasks_issue_count["easy"].append(task)
        elif issue_count <= 8:
            task["difficulty"] = "medium"
            tasks_issue_count["medium"].append(task)
        else:
            task["difficulty"] = "hard"
            tasks_issue_count["hard"].append(task)
    # Sample test set
    train_set = []
    test_set = []
    for difficulty, task_list in tasks_issue_count.items():
        if len(task_list) < 100:
            print(f"Warning: Not enough tasks for {difficulty} difficulty. Found {len(task_list)} tasks.")
            continue
        # Randomly sample 100 tasks for each difficulty
        sampled_tasks = random.sample(task_list, k=100)
        sampled_doc_ids = set(task["doc_id"] for task in sampled_tasks)
        test_set.extend(sampled_tasks)
        # Add the remaining tasks to the train set
        for task in task_list:
            if task["doc_id"] not in sampled_doc_ids:
                train_set.append(task)

    # Save the results
    with open("data/reasoning_tasks_train.jsonl", "w", encoding="utf-8") as f:
        for datum in train_set:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")
    with open("data/reasoning_tasks_test.jsonl", "w", encoding="utf-8") as f:
        for datum in test_set:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")
    
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

if __name__ == "__main__":
    main()
