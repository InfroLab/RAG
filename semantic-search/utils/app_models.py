from pydantic import BaseModel

class Text(BaseModel):
    text: str

class CandidateText(Text):
    text: str
    score: float