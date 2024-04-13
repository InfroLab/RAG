from fastapi import FastAPI
from .api.api import API
from .utils.app_models import Prompt, Response

api = API()
app = FastAPI()

@app.get("/answer-question")
async def answer_question(prompt: Prompt) -> Response:
    return await API.answer_question(prompt)