import random
import json
import csv
import os

random.seed(42)

def create_csv_from_json(json_data, output_filepath):
    """
    Creates a CSV file with data from a JSON list of dictionaries.

    Args:
        json_data (list): A list of dictionaries, where each dictionary
                          represents a document and contains a list of "issues".
        output_filepath (str): The full path where the CSV file will be saved.
    """
    all_rows = []
    headers = ["doc_id", "question", "difficulty", "model", "response", "issue"]
    all_rows.append(headers)

    for doc in json_data:
        doc_id = doc.get("doc_id", "")
        question = doc.get("question", "")
        difficulty = doc.get("difficulty", "")
        model = doc.get("model", "")
        response = doc.get("response", "")
        issues = doc.get("issues", [])

        if not issues:
            # If there are no issues, just add one row with the main info
            all_rows.append([doc_id, question, difficulty, model, response, ""])
        else:
            for i, issue in enumerate(issues):
                row = []
                # For the first issue, include all main document details
                if i == 0:
                    row.extend([doc_id, question, difficulty, model, response, issue])
                else:
                    # For subsequent issues, leave the main document cells blank
                    # to give the appearance of a multi-row cell
                    row.extend(["", "", "", "", "", issue])
                all_rows.append(row)

    try:
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(all_rows)
        print(f"CSV file successfully created at: {output_filepath}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")

def main():
    reasoning_tasks = {}
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        for line in f.readlines():
            task = json.loads(line)
            reasoning_tasks[task["doc_id"]] = task

    # Filter reasoning tasks that have more than 12 issues
    reasoning_tasks = {k: v for k, v in reasoning_tasks.items() if len(v["issues"]) <= 12}

    model_list = ["gemini-2.0-flash-001", "gemini-2.5-pro", "gemma-3-4b-it", "gemma-3-27b-it", "EXAONE-3.0-7.8B-Instruct", "exaone3.5:32b", "gpt-4.1-mini", "o3"]

    # Sample 15 reasoning tasks per each difficulty
    sampled_tasks = []
    for difficulty in ["easy", "medium", "hard"]:
        tasks = [task for task in reasoning_tasks.values() if task["difficulty"] == difficulty]
        sampled_tasks.extend(random.sample(tasks, 15))

    # Sample models to ensure each appears at least 5 times
    models = []
    # Determine how many times each model should appear
    min_appearances = 5
    num_models = len(model_list)
    total_tasks = len(sampled_tasks)

    # Calculate base distribution
    base_count = total_tasks // num_models
    remainder = total_tasks % num_models

    model_counts = {model: base_count for model in model_list}
    
    # Distribute the remainder
    for i in range(remainder):
        model_counts[model_list[i]] += 1

    # Ensure minimum appearances and adjust if necessary
    for model in model_list:
        if model_counts[model] < min_appearances:
            diff = min_appearances - model_counts[model]
            model_counts[model] = min_appearances
            # Try to take from models with more than base_count + 1
            candidates = [m for m, count in model_counts.items() if count > (base_count + (1 if m in model_list[:remainder] else 0))]
            while diff > 0 and candidates:
                for candidate_model in candidates:
                    if model_counts[candidate_model] > (base_count + (1 if candidate_model in model_list[:remainder] else 0)):
                        model_counts[candidate_model] -= 1
                        diff -= 1
                        if diff == 0:
                            break
                candidates = [m for m, count in model_counts.items() if count > (base_count + (1 if m in model_list[:remainder] else 0))]
            # If still need to reduce, take from any model with more than min_appearances
            while diff > 0:
                for candidate_model in model_list:
                    if model_counts[candidate_model] > min_appearances:
                        model_counts[candidate_model] -= 1
                        diff -= 1
                        if diff == 0:
                            break
    
    for model, count in model_counts.items():
        models.extend([model] * count)

    random.shuffle(models)
    
    assert len(models) == len(sampled_tasks), f"Mismatch in lengths: models={len(models)}, sampled_tasks={len(sampled_tasks)}"

    sampled_data = []
    for task, model in zip(sampled_tasks, models):
        # Open the specific model's results file
        results_file_path = f"results/reasoning_tasks_{model}.jsonl"
        
        # Check if the results file exists before trying to open it
        if not os.path.exists(results_file_path):
            print(f"Warning: Results file not found for model '{model}' at '{results_file_path}'. Skipping task {task['doc_id']}.")
            continue

        with open(results_file_path, "r", encoding="utf-8") as file:
            found_result = False
            for line in file.readlines():
                result = json.loads(line)
                if result["doc_id"] == task["doc_id"]:
                    # Reformat issues
                    issues = []
                    for i, issue in enumerate(task["issues"]):
                        issue_str = f"{i+1}. {issue['summary']}" # Start issue numbering from 1
                        if issue["id"] == "":
                            issue_str += "주문/청구취지"
                        for claim in issue["claim"]:
                            issue_str += f"\n    - {claim['claimer']}의 주장: {claim['content']}"
                        issue_str += f"\n    - 판사의 판단: {issue['conclusion']}"
                        issues.append(issue_str)
                    
                    sampled_data.append({
                        "doc_id": task["doc_id"],
                        "question": task["question"],
                        "model": model,
                        # "difficulty": task["difficulty"],
                        "response": result["response"],
                        "issues": issues,
                    })
                    found_result = True
                    break
            if not found_result:
                print(f"Warning: No matching result found for doc_id '{task['doc_id']}' in '{results_file_path}'.")


    print(json.dumps(sampled_data[0], ensure_ascii=False, indent=2))
    
    # Define the output directory and filename for the JSONL file
    output_jsonl_dir = "data"
    os.makedirs(output_jsonl_dir, exist_ok=True) # Create directory if it doesn't exist
    output_jsonl_filepath = os.path.join(output_jsonl_dir, "reasoning_tasks_humanjudgewithrubrics.jsonl")

    with open(output_jsonl_filepath, "w", encoding="utf-8") as f:
        for task in sampled_data:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"JSONL file successfully created at: {output_jsonl_filepath}")

    # Define the output directory and filename for the CSV file
    output_csv_dir = "data"
    os.makedirs(output_csv_dir, exist_ok=True) # Create directory if it doesn't exist
    output_csv_filepath = os.path.join(output_csv_dir, "rubric_based_judgment_issues.csv")

    create_csv_from_json(sampled_data, output_csv_filepath)

if __name__ == "__main__":
    main()