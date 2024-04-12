from typing import List, Dict

from pydantic import BaseModel

class Prompt(BaseModel):
    prompts: List[Dict[str,str]]

class QueryAndHistory(Prompt):
    query: str

class Text(BaseModel):
    text: str

class CandidateText(Text):
    text: str
    score: float