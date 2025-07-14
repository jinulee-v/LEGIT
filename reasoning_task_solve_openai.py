from openai import AsyncOpenAI

import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()

# Initialize OpenAI SDK
# models: gpt-4o-mini, gpt-4.1
MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1")
aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

api_cost = 0
default_generation_config = {
    # "max_output_tokens": 16384,
    "temperature": 1,
    "top_p": 0.95,
}
SYSTEM_PROMPT = "당신은 한국의 법률 전문가입니다. 주어진 사안과 청구취지를 잘 읽고 판결의 결과를 관련 법령/대법원 판례가 잘 드러나도록, 가능한 주장/항변/재항변 등을 폭넓게 검토한 뒤 판결의 결과를 예측하세요."
async def generate(prompt: str | list, response_schema=None):
    global api_cost
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if isinstance(prompt, list):
        messages.extend(prompt)
    else:
        messages.append({"role": "user", "content": prompt})

    response = await aclient.chat.completions.create(model=MODEL_NAME,
    messages=messages,
    # max_tokens=default_generation_config["max_output_tokens"],
    # temperature=default_generation_config["temperature"],
    # top_p=default_generation_config["top_p"],
    response_format={"type": "json_object"} if response_schema is not None else None)

    usage = response.usage
    api_cost += usage.prompt_tokens * 0.015 / 1000  # gpt-4o-mini price per 1K tokens (adjust as needed)
    api_cost += usage.completion_tokens * 0.03 / 1000

    try:
        if response_schema is not None:
            return response.choices[0].message.content.strip()
        else:
            return response.choices[0].message.content
    except Exception:
        return ""


async def main():
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        reasoning_tasks = [json.loads(line) for line in f.readlines()]

    results = []
    for task in reasoning_tasks:
        task_id = task["doc_id"]
        print(f"Processing task {task_id}...")

        # Convert legal issue to reasoning task
        prompt = SYSTEM_PROMPT + "\n\n" + task["question"]
        response = await generate(prompt)

        # print(response); exit()
        results.append({
            "doc_id": task_id,
            "response": response
        })

        # Save updated results
        with open(f"results/reasoning_tasks_{MODEL_NAME}.jsonl", "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
    print(f"API cost: ${api_cost:.6f}")
    print("Processing complete!")