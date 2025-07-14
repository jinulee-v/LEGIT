import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from pydantic import BaseModel

import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()

vertexai_generation_config = {
    "max_output_tokens": None,
    "temperature": 0.0
}

# Gemini setup

# Initialize Vertex AI
# models: gemini-2.0-flash-001, gemini-2.5-flash-001
PROJECT_ID=os.environ["PROJECT_ID"] # mandatory
REGION=os.environ.get("REGION", "us-west4") # default to us-west4
print(f"Initializing Vertex AI... PROJECT_ID: {PROJECT_ID}, REGION: {REGION}")
vertexai.init(project=PROJECT_ID, location=REGION)
print("Vertex AI initialization complete!")
    
api_cost = 0
default_generation_config = {
    "max_output_tokens": 500,
    "temperature": 1,
    "top_p": 0.95,
}
SYSTEM_PROMPT = "당신은 한국의 법률 전문가입니다. 주어진 사안과 청구취지를 잘 읽고 판결의 결과를 관련 법령/대법원 판례가 잘 드러나도록, 가능한 주장/항변/재항변 등을 폭넓게 검토한 뒤 판결의 결과를 예측하세요."
async def generate(model: GenerativeModel, prompt: str|list, response_schema=None):
    response = await model.generate_content_async(
        contents=prompt,
        generation_config=default_generation_config.copy().update({
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }) if (response_schema is not None) else default_generation_config
    )
    global api_cost
    api_cost += response.usage_metadata.prompt_token_count * 0.15/1000000
    api_cost += response.usage_metadata.candidates_token_count * 0.6/1000000
    try:
        if response_schema is not None:
            return response.text.replace("```json", "").replace("```", "").strip()
        else:
            return response.text
    except:
        return ""


async def main(args):
    with open("data/reasoning_tasks_test.jsonl", "r", encoding="utf-8") as f:
        reasoning_tasks = [json.loads(line) for line in f.readlines()]
    model = GenerativeModel(args.model)
    
    results = []
    for task in reasoning_tasks:
        task_id = task["doc_id"]
        print(f"Processing task {task_id}...")
        
        # Convert legal issue to reasoning task
        prompt = SYSTEM_PROMPT + "\n\n" + task["question"]
        response = await generate(model, prompt)

        # print(response); exit()
        results.append({
            "doc_id": task_id,
            "response": response
        })
        
        # Save updated results
        with open("results/reasoning_tasks_gemini2.5.jsonl", "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks using Vertex AI.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model name to use for evaluation.")
    args = parser.parse_args()

    asyncio.run(main(args))
    print(f"API cost: ${api_cost:.6f}")
    print("Processing complete!")