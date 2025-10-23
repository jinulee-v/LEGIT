import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import asyncio


from utils.router import get_model

load_dotenv()

SYSTEM_PROMPT = "당신은 한국의 법률 전문가입니다. 주어진 사안과 청구취지를 잘 읽고 판결의 결과를 관련 법령/대법원 판례가 잘 드러나도록, 가능한 주장/항변/재항변 등을 폭넓게 검토한 뒤 판결의 결과를 예측하세요."

async def main(args):
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        reasoning_tasks = [json.loads(line) for line in f.readlines()]
    retrieval_results = {} # doc_id -> [list of relevant laws] mapping
    with open(f"data/lawretrieval_test_{args.retrieval_method}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["doc_id"]
            related_laws = item["related_laws"]
            retrieval_results[doc_id] = related_laws
    with open("data/deduplicated_relevant_laws.json", "r", encoding="utf-8") as f:
        retrieval_corpora = {law["id"]: law["text"] for law in json.load(f)}
    for doc_id in retrieval_results:
        retrieval_results[doc_id] = [retrieval_corpora[law_id] for law_id in retrieval_results[doc_id] if law_id in retrieval_corpora]

    model_name_or_path = args.model
    model, generate = get_model(args.package, model_name_or_path)

    tasks = []
    for datum in reasoning_tasks:
        if "Qwen" in args.model:
            prompts = ["반드시 답변 전체를 **한국어**로만 작성하시오." + prompt for prompt in prompts]
        
        # prepend relevant law above the question
        prompt = ""
        relevant_laws = retrieval_results.get(datum["doc_id"], [])
        for law in relevant_laws:
            prompt += f"- {law}\n"
        
        prompt += "\n" + datum["question"]

        tasks.append(generate(model, prompt=prompt, system_prompt=SYSTEM_PROMPT))

    results = []
    for datum, output in zip(reasoning_tasks, await asyncio.gather(*tasks)):
        # output = await task
        results.append({
            "doc_id": datum["doc_id"],
            "response": output
        })

        # Save updated results
        model_name_or_path = model_name_or_path.split("/")[-1] # leave only the model name and remove org
        with open(f"results/reasoning_tasks_rag_{args.retrieval_method}_{model_name_or_path}.jsonl", "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks.")
    parser.add_argument("--package", type=str, help="package for LLM generation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Model name or path to use for evaluation.")
    parser.add_argument("--retrieval_method", type=str, default="groundtruth", help="Retrieval method used to get relevant laws.")
    args = parser.parse_args()

    import asyncio
    asyncio.run(main(args))
    print("Processing complete!")
