import ast
from typing import List, Dict

import torch

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

from ..utils.utils import load_tokenizer, read_model_name
from ..utils.app_models import Prompt, Response
from .config import LLMConfig

class LLM:
    def __init__(self, config: dict = None):
        self.config = LLMConfig(config)
        self.tokenizer, _, _ = load_tokenizer(self.config.tokenizer_dir)

    def parse_input(
                    self,
                    tokenizer,
                    input_text=None,
                    prompt_template=None,
                    add_special_tokens=True,
                    max_input_length=923,
                    pad_id=None,
                    num_prepend_vtokens=[],
                    model_name=None,
                    model_version=None
    ) -> List[torch.IntTensor]:
        if pad_id is None:
            pad_id = tokenizer.pad_token_id

        batch_input_ids = []
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text,
                                        add_special_tokens=add_special_tokens,
                                        truncation=True,
                                        max_length=max_input_length)
            batch_input_ids.append(input_ids)

        if num_prepend_vtokens:
            assert len(num_prepend_vtokens) == len(batch_input_ids)
            base_vocab_size = tokenizer.vocab_size - len(
                tokenizer.special_tokens_map.get('additional_special_tokens', []))
            for i, length in enumerate(num_prepend_vtokens):
                batch_input_ids[i] = list(
                    range(base_vocab_size,
                        base_vocab_size + length)) + batch_input_ids[i]

        if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
            for ids in batch_input_ids:
                ids.append(tokenizer.sop_token_id)

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]
        return batch_input_ids

    def beam_search(
                self,
                tokenizer,
                output_ids,
                input_lengths,
                sequence_lengths,
    ) -> List[str]:
        batch_size, num_beams, _ = output_ids.size()
        output_texts = []
        for batch_idx in range(batch_size):
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                output_texts.append(output_text)
        return output_texts

    def generate(self, prompt: str) -> List[str]:
        runtime_rank = tensorrt_llm.mpi_rank()
        logger.set_level(self.config.log_level)

        model_name, model_version = read_model_name(self.config.engine_dir)
        if self.config.tokenizer_dir is None:
            logger.warning(
                "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
            )
            raise ValueError('`tokenizer_dir` not specified.')

        tokenizer, pad_id, end_id = load_tokenizer(
            tokenizer_dir=self.config.tokenizer_dir,
            vocab_file=self.config.vocab_file,
            model_name=model_name,
            model_version=model_version,
            tokenizer_type=self.config.tokenizer_type,
        )

        stop_words_list = None
        bad_words_list = None

        prompt_template = None
        batch_input_ids = self.parse_input(tokenizer=tokenizer,
                                    input_text=prompt,
                                    prompt_template=prompt_template,
                                    input_file=self.config.input_file,
                                    add_special_tokens=self.config.add_special_tokens,
                                    max_input_length=self.config.max_input_length,
                                    pad_id=pad_id,
                                    num_prepend_vtokens=self.config.num_prepend_vtokens,
                                    model_name=model_name,
                                    model_version=model_version)
        input_lengths = [x.size(0) for x in batch_input_ids]

        if not PYTHON_BINDINGS and not self.config.use_py_session:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            self.config.use_py_session = True
        runner_cls = ModelRunner if self.config.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(engine_dir=self.config.engine_dir,
                            lora_dir=self.config.lora_dir,
                            rank=runtime_rank,
                            debug_mode=self.config.debug_mode,
                            lora_ckpt_source=self.config.lora_ckpt_source)
        if self.config.medusa_choices is not None:
            self.config.medusa_choices = ast.literal_eval(self.config.medusa_choices)
            assert self.config.use_py_session, "Medusa is only supported by py_session"
            assert self.config.temperature == 0, "Medusa should use temperature == 0"
            assert self.config.num_beams == 1, "Medusa should use num_beams == 1"
            runner_kwargs.update(medusa_choices=self.config.medusa_choices)
        if not self.config.use_py_session:
            runner_kwargs.update(
                max_batch_size=len(batch_input_ids),
                max_input_len=max(input_lengths),
                max_output_len=self.config.max_output_len,
                max_beam_width=self.config.num_beams,
                max_attention_window_size=self.config.max_attention_window_size,
                sink_token_length=self.config.sink_token_length,
            )
        runner = runner_cls.from_dir(**runner_kwargs)

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=self.config.max_output_len,
                max_attention_window_size=self.config.max_attention_window_size,
                sink_token_length=self.config.sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                repetition_penalty=self.config.repetition_penalty,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                lora_uids=self.config.lora_task_uids,
                prompt_table_path=self.config.prompt_table_path,
                prompt_tasks=self.config.prompt_tasks,
                streaming=self.config.streaming,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=self.config.medusa_choices)
            torch.cuda.synchronize()

        if runtime_rank == 0: # TODO
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']
            context_logits = None
            generation_logits = None
            if runner.gather_context_logits:
                context_logits = outputs['context_logits']
            if runner.gather_generation_logits:
                generation_logits = outputs['generation_logits']
            return self.beam_search(tokenizer,
                            output_ids,
                            input_lengths,
                            sequence_lengths,
                            output_csv=self.config.output_csv,
                            output_npy=self.config.output_npy,
                            context_logits=context_logits,
                            generation_logits=generation_logits,
                            output_logits_npy=self.config.output_logits_npy)

    def apply_template(self, prompt: Prompt, is_retrieval=True) -> torch.IntTensor:
        if is_retrieval:
            candidates_text = ''
            for idx, candidate in enumerate(prompt.candidates):
                candidates_text = candidates_text+f'{idx}. "{candidate.text}"\n'
            text_prompt = f"""
            Clippings from news articles for context:
            ---------------------
            {candidates_text}---------------------
            Given the context information and not prior knowledge, answer the query.
            Query: {prompt.query}
            Answer:
            """
            return self.tokenizer.apply_chat_template([{'roles':'user','content':text_prompt}], tokenize=False)
        else:
            history = prompt.history+[{'roles':'user','content':prompt.query}]
            return self.tokenizer.apply_chat_template(history, tokenize=False)
    
    def reply(self, prompt: Prompt, is_retrieval=True) -> Response:
        text_prompt = self.apply_template(prompt, is_retrieval)
        return Response(self.generate(text_prompt))