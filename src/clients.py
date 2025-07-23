import openai
import os


def get_llm_client():

    """"a simple function returning an OpenAI client"""

    client = openai.OpenAI(
        base_url="https://integrate.api.nvidia.com/v1", 
        api_key=os.getenv("NVIDIA_API_KEY") 
    )


    return client


def get_llm():

    """Just a simple function returnign the name of the used LLM"""

    model = "qwen/qwen3-235b-a22b"

    return model