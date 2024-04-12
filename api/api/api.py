from typing import List, Dict

import aiohttp

from ..utils.app_models import QueryAndHistory, CandidateText, Prompt

class API:
    def __init__(self):
        raise NotImplementedError
    
    async def find_candidates(self, qh: QueryAndHistory) -> List[CandidateText]:
        raise NotImplementedError
    
    def construct_text_prompt(self, query: str, candidates: List[CandidateText]) -> str:
        raise NotImplementedError

    async def generate_prompt(text_prompt: str, prompt: Prompt) -> Prompt:
        raise NotImplementedError
    
    async def answer_question(self, qh: QueryAndHistory) -> Prompt:
        raise NotImplementedError