import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import asyncio
import httpx

load_dotenv()

SYSTEM_PROMPT = "당신은 한국의 법률 전문가입니다. 주어진 사안과 청구취지를 잘 읽고 판결의 결과를 관련 법령/대법원 판례가 잘 드러나도록, 가능한 주장/항변/재항변 등을 폭넓게 검토한 뒤 판결의 결과를 예측하세요."
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

async def generate(model: str, prompt: str | list, response_schema=None):
    if isinstance(prompt, list):
        user_content = "\n".join([msg["content"] for msg in prompt if msg["role"] == "user"])
    else:
        user_content = prompt

    full_prompt = SYSTEM_PROMPT + "\n\n" + user_content

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

async def main(args):
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        reasoning_tasks = [json.loads(line) for line in f.readlines()]
    
    model_name = args.model

    results = []
    for task in reasoning_tasks:
        task_id = task["doc_id"]
        print(f"Processing task {task_id}...")
        prompt = task["question"]
        response = await generate(model_name, prompt)

        results.append({
            "doc_id": task_id,
            "response": response
        })

        # Save updated results
        with open(f"results/reasoning_tasks_{model_name}.jsonl", "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks using Ollama API.")
    parser.add_argument("--model", type=str, default="exaone3.5:7.8b", help="Model name to use for evaluation.")
    args = parser.parse_args()

    asyncio.run(main(args))
    print("Processing complete!")
