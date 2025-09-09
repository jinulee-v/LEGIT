import json
import argparse
import os

def evaluate(eval_file):
    results = {}

    # Compose doc_id to difficulty mapping
    doc_id_to_difficulty = {}
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        for line in f.readlines():
            task = json.loads(line)
            doc_id_to_difficulty[task["doc_id"]] = task["difficulty"]
        

    with open(eval_file, "r", encoding="utf-8") as f:
        reasoning_tasks = [json.loads(line) for line in f.readlines()]

    # Compute and print statistics
    num_tasks = len(reasoning_tasks)
    # print(f"Total number of reasoning tasks: {num_tasks}")

    # Compute average scores per each difficulty level
    difficulty_scores = {"easy": [], "medium": [], "hard": []}
    for task in reasoning_tasks:
        difficulty = doc_id_to_difficulty.get(task["doc_id"])
        score = task.get("score", 0)
        if difficulty not in difficulty_scores:
            difficulty_scores[difficulty] = []
        difficulty_scores[difficulty].append(score)
    
    for difficulty, scores in difficulty_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        results["score/" + difficulty] = avg_score
        # print(f"Average score for {difficulty} tasks: {avg_score:.2f}")
    results["score/total"] = (results["score/easy"] + results["score/medium"] + results["score/hard"]) / 3

    # Print the root accuracy (issue_id == "") per difficulty
    root_accuracy = {"easy": {"correct": 0, "total": 0}, "medium": {"correct": 0, "total": 0}, "hard": {"correct": 0, "total": 0}}
    for task in reasoning_tasks:
        difficulty = doc_id_to_difficulty.get(task["doc_id"])
        if difficulty not in root_accuracy:
            root_accuracy[difficulty] = {"correct": 0, "total": 0}
        
        if "" not in task["evaluation_result"]:
            # print("Something wrong")
            continue
        root_accuracy[difficulty]["total"] += 1
        if task["evaluation_result"][""]["correct_conclusion"]:
            root_accuracy[difficulty]["correct"] += 1
    
    for difficulty, stats in root_accuracy.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        results["root_accuracy/" + difficulty] = accuracy
        # print(f"Root accuracy for {difficulty} tasks: {accuracy:.2f} ({stats['correct']} correct out of {stats['total']} total)")
    results["root_accuracy/total"] = (results["root_accuracy/easy"] + results["root_accuracy/medium"] + results["root_accuracy/hard"]) / 3

    return results

def main():
    results = {}
    for file in os.listdir("results"):
        if "evaluator_gemini-2.5-flash" in file:
            # skip gemma-3-27b-it
            eval_file = os.path.join("results", file)
            generator_model = "_evaluator_".join(file.split("_evaluator_")[:-1]).split("reasoning_tasks_")[1]
            print(f"Evaluating {eval_file} ({generator_model})...")
            results[generator_model] = evaluate(eval_file)
            print("\n")
    
    print("===============")

    # Sort by "score/total" and print the results
    sorted_results = sorted(results.items(), key=lambda x: x[1]["score/total"], reverse=True)
    print("Total score:")
    for model, stats in sorted_results:
        print(f"  {model}: {stats['score/total']:.2f}")
    print("==================")
    
    # gemma-3 scores/root correct by difficulty level
    print("gemma-3-4b-it")
    gemma_results = results.get("gemma-3-4b-it", {})
    print(f"  score/easy: {gemma_results['score/easy']:.2f}, score/medium: {gemma_results['score/medium']:.2f}, score/hard: {gemma_results['score/hard']:.2f}")
    print(f"  root_accuracy/easy: {gemma_results['root_accuracy/easy']:.2f}, root_accuracy/medium: {gemma_results['root_accuracy/medium']:.2f}, root_accuracy/hard: {gemma_results['root_accuracy/hard']:.2f}")

    print("gemma-3-12b-it")
    gemma_results = results.get("gemma-3-12b-it", {})
    print(f"  score/easy: {gemma_results['score/easy']:.2f}, score/medium: {gemma_results['score/medium']:.2f}, score/hard: {gemma_results['score/hard']:.2f}")
    print(f"  root_accuracy/easy: {gemma_results['root_accuracy/easy']:.2f}, root_accuracy/medium: {gemma_results['root_accuracy/medium']:.2f}, root_accuracy/hard: {gemma_results['root_accuracy/hard']:.2f}")

    print("gemma-3-27b-it")
    gemma_results = results.get("gemma-3-27b-it", {})
    print(f"  score/easy: {gemma_results['score/easy']:.2f}, score/medium: {gemma_results['score/medium']:.2f}, score/hard: {gemma_results['score/hard']:.2f}")
    print(f"  root_accuracy/easy: {gemma_results['root_accuracy/easy']:.2f}, root_accuracy/medium: {gemma_results['root_accuracy/medium']:.2f}, root_accuracy/hard: {gemma_results['root_accuracy/hard']:.2f}")

    
    print("gemma3-4b_evaluator_gemma3-27b_fullreward")
    gemma_results = results.get("gemma3-4b_evaluator_gemma3-27b_fullreward", {})
    print(f"  score/easy: {gemma_results['score/easy']:.2f}, score/medium: {gemma_results['score/medium']:.2f}, score/hard: {gemma_results['score/hard']:.2f}")
    print(f"  root_accuracy/easy: {gemma_results['root_accuracy/easy']:.2f}, root_accuracy/medium: {gemma_results['root_accuracy/medium']:.2f}, root_accuracy/hard: {gemma_results['root_accuracy/hard']:.2f}")
    
    print("gemma3-4b_evaluator_gemma3-27b_rootreward")
    gemma_results = results.get("gemma3-4b_evaluator_gemma3-27b_rootreward", {})
    print(f"  score/easy: {gemma_results['score/easy']:.2f}, score/medium: {gemma_results['score/medium']:.2f}, score/hard: {gemma_results['score/hard']:.2f}")
    print(f"  root_accuracy/easy: {gemma_results['root_accuracy/easy']:.2f}, root_accuracy/medium: {gemma_results['root_accuracy/medium']:.2f}, root_accuracy/hard: {gemma_results['root_accuracy/hard']:.2f}")


if __name__ == "__main__":
    main()