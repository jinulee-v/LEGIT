import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
from pydantic import BaseModel

import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import re
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
    "temperature": 0,
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


def is_leaf(issue, datum):
    for x in datum["issues"]:
        if x["id"] != issue["id"] and x["id"].startswith(issue["id"]):
            return False
    return True

summary_pattern = re.compile(r".*의 .*(주장|청구)에 대한 판단")
supremecourt_citation_pattern = re.compile(r"(.*?\(대법원.*?참조\))[.,]?")
complex_conclusion = re.compile(r"(에 따르면|에 따라|이므로|하므로|으므로|인 바,|한 바|따라서|\. )")

async def process_datum(datum, prompts):
    # 1. If event summary is exactly from the prompt, re-extract.
    if "코뼈골절" in datum["event_summary"] or len(datum["events"]) == 0:
        events = await generate(
            prompt=prompts["extract_events_to_json"].format(precedent=datum["precedent"]),
            response_schema=list[Event],
        )
        event_summary = await generate(
            prompt=prompts["summarize_events"].format(events=events)
        )
        try:
            datum["events"] = json.loads(events)
            datum["event_summary"] = event_summary
            if "코뼈골절" in datum["event_summary"]:
                print("Fail to generate correct events")
            else:
                print("Success!")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON (events): {e}")
            # print(events)
            datum["events"] = []
    
    # 2. If Issue's claim consists of multiple claims, split them.   
    # Extract issues
    new_issues = []
    for issue_idx in range(len(datum["issues"])):
        issue = datum["issues"][issue_idx]
        # relevant_law - split supreme court citations
        new_law = []
        for law in issue["relevant_law"]:
            new_law.extend(re.split(supremecourt_citation_pattern, law))
        issue["relevant_law"] = [x.strip() for x in new_law if len(x.strip()) > 0]
        # claim:
        # 1. merge claims from the same claimer
        new_claim = {}
        for claim in issue["claim"]:
            if claim["claimer"] not in new_claim:
                new_claim[claim["claimer"]] = claim["content"]
            else:
                new_claim[claim["claimer"]] += " " + claim["content"]
        issue["claim"] = [{"claimer": k, "content": v} for k, v in new_claim.items()]
        # 2. If claim is complex, split it to multiple claims
        if is_leaf(issue, datum) and any([len(x["content"].split("다. ")) >= 2 for x in issue["claim"]]):
            # Split issue to subissues
            # print(f"Patch claim: {issue['id']}")
            # print(json.dumps(issue, ensure_ascii=False, indent=2))
            new_issue = await generate(
                prompts["patch_claim_split"].format(issue=json.dumps(issue, ensure_ascii=False)),
                response_schema=list[Issues]
            )
            # print(new_issue)
            # print("-" * 30)
            try:
                new_issues.extend(json.loads(new_issue))
            except json.JSONDecodeError as e:
                new_issues.append(issue)
        else:
            # If not complex, just keep the issue
            new_issues.append(issue)
    datum["issues"] = new_issues

    # If two issues of the same level share same law and analysis, merge them to the parent node.
    law_and_analysis_to_issue_ids = {}
    for issue in datum["issues"]:
        key = (tuple(issue["analysis"]))
        if key not in law_and_analysis_to_issue_ids:
            law_and_analysis_to_issue_ids[key] = []
        law_and_analysis_to_issue_ids[key].append(issue["id"])
    # Merge issues
    for issue_ids in law_and_analysis_to_issue_ids.values():
        if len(issue_ids) > 1 and \
            len(set([len(issue_id.split(".")) for issue_id in issue_ids])) == 1 and \
            len(set([issue_id.rsplit(".", 1)[0] for issue_id in issue_ids])) == 1:
            print(datum['doc_id'], issue_ids)
    return datum

async def main_async():
    # DEBUG
    # data_corpus = data_corpus[0:30]
    # Resume training
    results = []
    data_corpus = []
    if os.path.exists("data/legit.jsonl"):
        completed_doc_ids = set()
        with open("data/legit.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                datum = json.loads(line)
                data_corpus.append(datum)

        print(f"Resuming from {len(completed_doc_ids)} docs...")
        print(f"Total docs to patch: {len(data_corpus)}")


    # Parallel processing
    BATCH_SIZE = 4
    batch = []
    for i in range(0, len(data_corpus)):
        datum = data_corpus[i]
        if ("코뼈골절" not in datum["event_summary"]) and len(datum["issues"]) > 0 and len(datum["events"]) > 0:
            continue
        batch.append(data_corpus[i])
        if len(batch) < BATCH_SIZE and i < len(data_corpus) - 1:
            continue
        print([datum["doc_id"] for datum in batch])

        # Process batch
        tasks = []
        print(f"Creating coroutines...")
        for datum in batch:
            tasks.append(asyncio.create_task(process_datum(datum, prompts)))
        print("Receiving results...")
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await f
            batch[batch.index(result)] = result
            # results.append(result)
        # Save the results
        with open("data/legit.jsonl", "w", encoding="utf-8") as f:
            for datum in data_corpus:
                f.write(json.dumps(datum, ensure_ascii=False) + "\n")
        batch = []

    print("API cost:", api_cost)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
