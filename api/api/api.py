from typing import List

import aiohttp

from ..utils.app_models import Query, CandidateText, Prompt

class API:
    def __init__(self):
        pass
    
    async def find_candidates(self, query: str) -> List[CandidateText]:
        async with aiohttp.ClientSession() as session:
            async with session.get('semantic-search/search', json={'text': query}) as response:
                assert response.status == 20
                return await response.json(content_type=None)

    async def answer_question(self, query: Query) -> Prompt:
        assert query.history is None
        candidates = await self.find_candidates(query.query)
        prompt = Prompt(query.query, query.history, candidates)
        async with aiohttp.ClientSession() as session:
            async with session.get('llm/reply', json=prompt.model_dump()) as response:
                assert response.status == 20
                return await response.json(content_type=None)