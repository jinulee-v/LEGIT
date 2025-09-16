
from typing import Tuple, Any, Callable
import os

def get_model(package: str, model_name: str) -> Tuple[Any, Callable]:
    if package == "vertexai":
        # assert model_name in ["gemini-2.0-flash-001", "gemini-2.5-flash-001", "gemini-2.5-pro"]
        import vertexai
        from vertexai.preview.generative_models import GenerativeModel
        from utils.packages.vertexai import generate
        
        PROJECT_ID=os.environ["PROJECT_ID"] # mandatory
        REGION=os.environ.get("REGION", "us-west4") # default to us-west4
        print(f"Initializing Vertex AI... PROJECT_ID: {PROJECT_ID}, REGION: {REGION}")
        vertexai.init(project=PROJECT_ID, location=REGION)
        print("Vertex AI initialization complete!")

        return GenerativeModel(model_name), generate
    
    if package == "ollama":
        # assert model_name in ["exaone3.5:7.8b", "exaone3.5:13b"]
        from utils.packages.ollama import generate
        return model_name, generate

    if package == "vllm":
        # assert model_name in ["qwen2.5-72b", "qwen2.5-14b", "qwen2.5-7b"]
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        from utils.packages.vllm import generate
        
        args = AsyncEngineArgs(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=0.8, trust_remote_code=True)
        llm = AsyncLLM.from_engine_args(args)
        return llm, generate

    if package == "openai":
        # assert model_name in ["gpt-3.5-turbo", "gpt-4"]
        from utils.packages.openai import generate
        return model_name, generate
