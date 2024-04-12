from fastapi import FastAPI

from .utils.app_models import Prompt, Response
from .llm.llm import LLM

llm = LLM()
app = FastAPI()

@app.get("/reply")
async def reply(prompt: Prompt):
    return Response(llm.generate(prompt))