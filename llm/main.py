from fastapi import FastAPI

from .utils.app_models import Prompt, Response
from .llm.llm import LLM

llm = LLM({"tokenizer_dir": "../model/mistral-7b-v0.2-trtllm-int4/Mistral-7B-Instruct-v0.2", "engine_dir": '../engine'})
app = FastAPI()

@app.get("/reply")
async def reply(prompt: Prompt) -> Response:
    return llm.reply(prompt)