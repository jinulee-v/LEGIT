
from utils.router import get_model
from utils.prompts import prompts

from pydantic import BaseModel

import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import asyncio
import argparse
load_dotenv()

class Evaluation(BaseModel):
    rationales: str
    contains_issue: bool
    correct_conclusion: bool

async def main(args):
    path = args.response_path
    model, generate = get_model(args.package, args.model)

    reasoning_tasks = {} # "id" -> object
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        for line in f.readlines():
            task = json.loads(line)
            reasoning_tasks[task["doc_id"]] = task

    results = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                result = json.loads(line)
                results.append(result)

    for result in results:
        task_id = result["doc_id"]
        response = result["response"]
        reasoning_task = reasoning_tasks[task_id]
        print(f"Processing task {task_id}...")
        # print("RESPONSE:\n", response)
        
        score = 0

        # Evaluate issues
        issue_results = {}
        for issue in reasoning_task["issues"]:
            # Check if the response contains the issue
            try:
                evaluate_raw = await generate(model, prompt=prompts["evaluate_issue_in_response"].format(
                    response=response,
                    summary=issue["summary"],
                    claims="\n".join([f"{claim['claimer']}: {claim['content']}" for claim in issue["claim"]]),
                    conclusion=issue["conclusion"],
                ), system_prompt="", response_schema=Evaluation)
                evaluate_raw = evaluate_raw.replace("<OUTPUT>", "").replace("</OUTPUT>", "").strip()
            except ValueError:
                print("Error (length?)")
                continue
            # print("----------------------------")

            # print(response)
            # print("EVALUATING ISSUE:", issue["id"])
            # print("SUMMARY:", issue["summary"])
            # print("CLAIMS:\n", "\n".join([f"{claim['claimer']}: {claim['content']}" for claim in issue["claim"]]))
            # print("CONCLUSION:\n", issue["conclusion"])
            # print("EVALUATE RESULT:\n", evaluate_raw)
            try:
                # evaluate = evaluate_raw.replace("```json", "").replace("```", "").strip()
                # evaluate = evaluate.split("<JSON>")[-1].split("</JSON>")[0].strip()
                evaluate = json.loads(evaluate_raw)
                contains_issue, correct_conclusion = evaluate["contains_issue"], evaluate["correct_conclusion"]
                issue_results[issue["id"]] = {
                    "rationales": evaluate["rationales"],
                    "contains_issue": contains_issue,
                    "correct_conclusion": correct_conclusion
                }
            except json.JSONDecodeError:
                print("Error decoding JSON from evaluation response. Skipping this issue.")
                continue
        
        for issue_id, issue_result in issue_results.items():
            # Check if any of the subissue have contains_issue as True and correct_conclusion as False
            subissue_fulfilled = True
            for sub_issue_id, sub_issue_result in issue_results.items():
                if sub_issue_id.startswith(issue_id) and issue_id != sub_issue_id:
                    if sub_issue_result["contains_issue"] and not sub_issue_result["correct_conclusion"]:
                        subissue_fulfilled = False
                        issue_result["score"] = 0
                        break
            issue_score = 0
            if issue_id == "":
                # if conclusion is correct, score 5
                if issue_result["contains_issue"] and issue_result["correct_conclusion"] and subissue_fulfilled:
                    issue_score += 5
            else:
                if issue_result["contains_issue"] and subissue_fulfilled:
                    issue_score += 2 / (len(issue_results) - 1)
                    if issue_result["correct_conclusion"]:
                        issue_score += 3 / (len(issue_results) - 1)
            issue_result["score"] = issue_score
            score += issue_score


        # Add the score to the result
        print("SCORE:", score)
        result["evaluator_model"] = args.model
        result["evaluation_result"] = issue_results
        result["score"] = score
        model_alias = args.model.split("/")[-1]
        with open(path.replace(".jsonl", f"_evaluator_{model_alias}.jsonl"), "w", encoding="utf-8") as f:
            for result in results:
                if "evaluation_result" in result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks using Vertex AI.")
    parser.add_argument("--response_path", type=str, required=True, help="Path to the JSONL file containing LLM responses.")
    parser.add_argument("--package", type=str, required=True, help="Package for LLM generation.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-001", help="Model name to use for evaluation.")
    args = parser.parse_args()

    asyncio.run(main(args))
    # print(f"API cost: ${api_cost:.6f}")
    print("Processing complete!")
