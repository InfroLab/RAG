from fastapi import FastAPI
from .api.api import API
from .utils.app_models import QueryAndHistory

api = API()
app = FastAPI()

@app.get("/answer-question")
async def answer_question(qh: QueryAndHistory):
    return await API.answer_question(qh)