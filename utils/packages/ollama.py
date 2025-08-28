import os
import json
from dotenv import load_dotenv
import httpx
import asyncio

load_dotenv()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

async def generate(model: str, prompt: str, system_prompt: str, response_schema=None, ollama_url=OLLAMA_URL) -> str:
    user_content = system_prompt + "\n\n" + prompt
    payload = {
        "model": model,
        "prompt": user_content,
        "stream": False
    }
    if response_schema:
        payload["format"] = response_schema.model_json_schema()

    async with httpx.AsyncClient() as client:
        responses = await client.post(ollama_url, json=payload, timeout=300)
    if response_schema is None:
        return responses.json().get("response", "").strip()
    else:
        return json.loads(responses.json().get("response", "").strip())
