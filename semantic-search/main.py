from typing import List

from fastapi import FastAPI
from fastapi import status

from .utils.app_models import Text, CandidateText
from .retreiver.retriever import VectorStoreRetriever

retriever = VectorStoreRetriever()
app = FastAPI()

@app.get("/generate-index")
async def generate_index():
    retriever.generate_index()
    return status.HTTP_200_OK

@app.get("/generate-embedding")
async def generate_embedding():
    return status.HTTP_404_NOT_FOUND 

@app.get("/add/question")
async def add_question(text: Text):
    return status.HTTP_404_NOT_FOUND 

@app.get("/search")
async def search(text: Text) -> List[CandidateText]:
    nodes = retriever.retrieve(text.text)
    candidates = []
    for node in nodes:
        candidates.append(CandidateText(node.text, node.score))
    return candidates