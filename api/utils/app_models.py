from typing import List, Dict, Optional

from typing import Optional
from pydantic import BaseModel, validator

class CandidateText(BaseModel):
    text: str
    score: float

class Query(BaseModel):
    query: str
    history: Optional[List[Dict[str,str]]]

class Response(BaseModel):
    text: List[str]

class Prompt(Query):
    candidates: Optional[List[CandidateText]]

    @validator("candidates", always=True)
    def mutually_exclusive(cls, v, values):
        if values["history"] and v:
            raise ValueError("'history' and 'candidates' are mutually exclusive.")
        return v