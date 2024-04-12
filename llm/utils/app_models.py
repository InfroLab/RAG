from typing import List, Dict

from pydantic import BaseModel

class Prompt(BaseModel):
    prompts: List[Dict[str,str]]

class Response(BaseModel):
    text: str