from ..utils.app_models import Prompt

class LLM:
    def __init__(self):
        raise NotImplementedError
    
    def process_prompt(self, prompt: Prompt) -> str:
        raise NotImplementedError

    def generate(self, text_prompt: str) -> str:
        raise NotImplementedError