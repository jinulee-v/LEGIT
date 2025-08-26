import vertexai
from vertexai.preview.generative_models import GenerativeModel

import os
from dotenv import load_dotenv
import json
load_dotenv()

vertexai_generation_config = {
    "max_output_tokens": None,
    "temperature": 0.0
}
# Gemini setup

# Initialize Vertex AI
PROJECT_ID=os.environ["PROJECT_ID"] # mandatory
REGION=os.environ.get("REGION", "us-west4") # default to us-west4
print(f"Initializing Vertex AI... PROJECT_ID: {PROJECT_ID}, REGION: {REGION}")
vertexai.init(project=PROJECT_ID, location=REGION)
print("Vertex AI initialization complete!")
    
api_cost = 0
default_generation_config = {
    "max_output_tokens": 50000,
    "temperature": 0
}

async def generate(model: GenerativeModel, prompt: str, system_prompt: str, response_schema=None):
    response = await model.generate_content_async(
        contents=system_prompt + "\n\n" + prompt,
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
            response_text = response.text.split("<OUTPUT>")[-1].split("</OUTPUT>")[0]
            return json.loads(response_text.replace("```json", "").replace("```", "").strip())
        else:
            return response.text
    except Exception as e:
        print(e.__class__, e)
        return None