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

class Event(BaseModel):
    event: str

class Claim(BaseModel):
    claimer: str
    content: str
class Issues(BaseModel):
    id: str
    summary: str
    claim: list[Claim]
    relevant_law: list[str]
    analysis: list[str]
    result: str
    
vertexai_generation_config = {
    "max_output_tokens": None,
    "temperature": 0.0
}

# Load prompts
prompts = {x: None for x in ["extract_events_to_json", "extract_issues_to_json_2shot", "summarize_events", "extract_issues_self_refine"]}
for prompt in prompts.keys():
    with open(f"prompts/{prompt}.txt", "r", encoding="utf-8") as f:
        prompts[prompt] = f.read()

# Gemini setup

# Initialize Vertex AI
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
PROJECT_ID=os.environ["PROJECT_ID"] # mandatory
REGION=os.environ.get("REGION", "us-west4") # default to us-west4
print(f"Initializing Vertex AI... PROJECT_ID: {PROJECT_ID}, REGION: {REGION}")
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel(MODEL_NAME)
print("Vertex AI initialization complete!")
    
api_cost = 0
default_generation_config = {
    "max_output_tokens": 50000,
    "temperature": 1,
    "top_p": 0.95,
}
async def generate(prompt: str|list, response_schema=None):
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



async def process_datum(datum, prompts):
    # 1. Extract events and summarize
    events = await generate(
        prompt=prompts["extract_events_to_json"].format(precedent=datum["precedent"]),
        response_schema=list[Event],
    )
    event_summary = await generate(
        prompt=prompts["summarize_events"].format(events=events)
    )
    # 2. Extract issues
    issues = await generate(
        prompt=prompts["extract_issues_to_json_2shot"].format(precedent=datum["precedent"]),
        response_schema=list[Issues],
    )
    refined_issues = await generate(
        # prompt=[
        #     {"role": "user", "parts": [{"text": prompts["extract_issues_to_json_2shot"].format(precedent=datum["precedent"])}]},
        #     {"role": "assistant", "parts": [{"text": issues}]},
        #     {"role": "user", "parts": [{"text": prompts["extract_issues_self_refine"]}]}
        # ],
        prompt = "<USER>" + prompts["extract_issues_to_json_2shot"].format(precedent=datum["precedent"]) + "</USER>" \
                 "<ASSISTANT>" + issues + "</ASSISTANT>" \
                 "<USER>" + prompts["extract_issues_self_refine"] + "</USER>",
        response_schema=list[Issues],
    )
    datum["event_summary"] = event_summary
    try:
        datum["events"] = json.loads(events)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON (events): {e}")
        # print(events)
        datum["events"] = []
    try:
        datum["issues"] = json.loads(refined_issues)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON (issues): {e}")
        # print(refined_issues)
        datum["issues"] = []
    # print(datum["precedent"])
    # print("Issues:", json.dumps(datum["issues"], ensure_ascii=False, indent=2))
    # print("\n" + "="*50 + "\n")
    return datum

async def main_async():
    # Load the dataset
    data_corpus = []
    with open("data/precedents.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data_corpus.append(json.loads(line))
    # DEBUG
    # data_corpus = data_corpus[0:30]
    # Resume training
    results = []
    if os.path.exists("data/legit.jsonl"):
        completed_doc_ids = set()
        with open("data/legit.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                datum = json.loads(line)
                completed_doc_ids.add(datum["doc_id"])
                results.append(datum)
        if len(completed_doc_ids) > 0:
            print(f"Resuming from {len(completed_doc_ids)} docs...")
            data_corpus = [datum for datum in data_corpus if datum["doc_id"] not in completed_doc_ids]
            print(f"Total docs after removing duplicates: {len(data_corpus)}")


    # Parallel processing
    BATCH_SIZE = 32
    for i in range(0, len(data_corpus), BATCH_SIZE):
        end_i = min(i + BATCH_SIZE, len(data_corpus))
        batch = data_corpus[i:end_i]

        tasks = []
        print(f"Creating coroutines ({i+1}-{end_i} / {len(data_corpus)})...")
        for datum in batch:
            tasks.append(asyncio.create_task(process_datum(datum, prompts)))
        print("Receiving results...")
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await f
            results.append(result)
        # Save the results
        with open("data/legit.jsonl", "w", encoding="utf-8") as f:
            for datum in results:
                f.write(json.dumps(datum, ensure_ascii=False) + "\n")

    print("API cost:", api_cost)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
