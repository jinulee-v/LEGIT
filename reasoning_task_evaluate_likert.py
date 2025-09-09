
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
    score: int

# Load LEGIT dataset
legit_doc_id_to_judgment = {}
with open("data/legit.jsonl", "r", encoding="utf-8") as f:
    for line in f.readlines():
        judgment = json.loads(line)
        legit_doc_id_to_judgment[judgment["doc_id"]] = judgment["precedent"]

async def main(args):
    path = args.response_path
    model, generate = get_model(args.package, args.model)
    print("Reasoning task evaluiate script", args.response_path, args.model)

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
        print(f"Processing task {task_id}...")
        # print("RESPONSE:\n", response)
        
        score = 0

        try:
            evaluate = await generate(model, prompt=prompts["evaluate_likert_scale"].format(
                judgment=legit_doc_id_to_judgment[task_id],
                response=response,
            ), system_prompt="", response_schema=Evaluation)
            score = evaluate["score"]
        except Exception as e:
            print(f"Error: {e.__class__} {e}")
            score = 0
            continue

        result["score"] = score
        model_alias = args.model.split("/")[-1]
        with open(path.replace(".jsonl", f"_likertevaluator_{model_alias}.jsonl"), "w", encoding="utf-8") as f:
            for result in results:
                if "score" in result:
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
