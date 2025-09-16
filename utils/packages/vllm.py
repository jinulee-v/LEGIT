import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
import asyncio
from vllm import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import GuidedDecodingParams

load_dotenv()

default_generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_output_tokens": 16384,  # Adjusted for vLLM
}
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:11434/api/generate")
request_id_cnt = 0
lock = asyncio.Lock()

async def generate(model: AsyncLLM, prompt: str, system_prompt: str, response_schema=None, vllm_url=VLLM_URL):
    prompt = system_prompt + "\n\n" + prompt
    if response_schema is not None:
        guided_decoding_params = GuidedDecodingParams(
            json=response_schema.model_json_schema()
        )
    else:
        guided_decoding_params = None
    sampling_params = SamplingParams(
        temperature=default_generation_config["temperature"],
        top_p=default_generation_config["top_p"],
        max_tokens=default_generation_config["max_output_tokens"],
        guided_decoding=guided_decoding_params,
    )
    async with lock:
        global request_id_cnt
        request_id = f"vllm_async_request_{request_id_cnt}"
        request_id_cnt += 1
    results_generator = model.generate(prompt, sampling_params, request_id=request_id)
    output = None
    async for request_output in results_generator:
        output = request_output
    if response_schema is not None:
        return json.loads(output.outputs[0].text.strip())
    else:
        return output.outputs[0].text.strip()
