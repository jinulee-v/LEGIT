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
aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

api_cost = 0
default_generation_config = {
    # "max_output_tokens": 16384,
    "temperature": 1,
    "top_p": 0.95,
}

async def generate(model: str, prompt: str, system_prompt: str, response_schema=None):
    global api_cost
    messages = [{"role": "system", "content": system_prompt}]
    if isinstance(prompt, list):
        messages.extend(prompt)
    else:
        messages.append({"role": "user", "content": prompt})

    response = await aclient.chat.completions.create(model=model,
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
            return json.loads(response.choices[0].message.content.strip())
        else:
            return response.choices[0].message.content
    except Exception:
        return None
