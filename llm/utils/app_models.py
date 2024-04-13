from typing import List, Dict, Optional

from typing import Optional
from pydantic import BaseModel, validator

class CandidateText(BaseModel):
    text: str
    score: float

class Prompt(BaseModel):
    query: str
    history: Optional[List[Dict[str,str]]]
    candidates: Optional[List[CandidateText]]

    @validator("candidates", always=True)
    def mutually_exclusive(cls, v, values):
        if values["history"] and v:
            raise ValueError("'history' and 'candidates' are mutually exclusive.")
        return v

class Response(BaseModel):
    text: List[str]