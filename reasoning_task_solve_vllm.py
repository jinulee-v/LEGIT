import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import asyncio
from vllm import LLM, SamplingParams

load_dotenv()

default_generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_output_tokens": 2048,  # Adjusted for vLLM
}
SYSTEM_PROMPT = "당신은 한국의 법률 전문가입니다. 주어진 사안과 청구취지를 잘 읽고 판결의 결과를 관련 법령/대법원 판례가 잘 드러나도록, 가능한 주장/항변/재항변 등을 폭넓게 검토한 뒤 판결의 결과를 예측하세요."

def build_prompt(task):
    return SYSTEM_PROMPT + "\n\n" + task["question"]

async def main(args):
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        reasoning_tasks = [json.loads(line) for line in f.readlines()]

    model_name_or_path = args.model
    llm = LLM(model=model_name_or_path)
    sampling_params = SamplingParams(
        temperature=default_generation_config["temperature"],
        top_p=default_generation_config["top_p"],
        max_tokens=default_generation_config["max_output_tokens"],
    )

    batch_size = 8  # You can adjust this
    results = []

    for i in tqdm(range(0, len(reasoning_tasks), batch_size)):
        batch_tasks = reasoning_tasks[i:min(i+batch_size, len(reasoning_tasks))]
        prompts = [build_prompt(task) for task in batch_tasks]

        outputs = llm.generate(prompts, sampling_params)
        for task, output in zip(batch_tasks, outputs):
            results.append({
                "doc_id": task["doc_id"],
                "response": output.outputs[0].text.strip()
            })

        # Save updated results
        with open(f"results/reasoning_tasks_{model_name_or_path}.jsonl", "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks using vLLM.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Model name or path to use for evaluation.")
    args = parser.parse_args()

    # vLLM is not async, so just call main synchronously
    import asyncio
    asyncio.run(main(args))
    print("Processing complete!")
