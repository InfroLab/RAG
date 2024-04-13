import argparse
from argparse import Namespace

class LLMConfig:
    def __init__(self, update_dict: dict = None) -> Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_output_len', type=int, required=True)
        parser.add_argument(
            '--max_attention_window_size',
            type=int,
            default=None,
            help=
            'The attention window size that controls the sliding window attention / cyclic kv cache behaviour'
        )
        parser.add_argument('--sink_token_length',
                            type=int,
                            default=None,
                            help='The sink token length.')
        parser.add_argument('--log_level', type=str, default='error')
        parser.add_argument('--engine_dir', type=str, default='engine_outputs')
        parser.add_argument('--use_py_session',
                            default=False,
                            action='store_true',
                            help="Whether or not to use Python runtime session")
        parser.add_argument(
            '--no_prompt_template',
            dest='use_prompt_template',
            default=True,
            action='store_false',
            help=
            "Whether or not to use default prompt template to wrap the input text.")
        parser.add_argument(
            '--input_file',
            type=str,
            help=
            'CSV or Numpy file containing tokenized input. Alternative to text input.',
            default=None)
        parser.add_argument('--max_input_length', type=int, default=923)
        parser.add_argument('--output_csv',
                            type=str,
                            help='CSV file where the tokenized output is stored.',
                            default=None)
        parser.add_argument('--output_npy',
                            type=str,
                            help='Numpy file where the tokenized output is stored.',
                            default=None)
        parser.add_argument(
            '--output_logits_npy',
            type=str,
            help=
            'Numpy file where the generation logits are stored. Use only when num_beams==1',
            default=None)
        parser.add_argument('--tokenizer_dir',
                            help="HF tokenizer config path",
                            default='gpt2')
        parser.add_argument(
            '--tokenizer_type',
            help=
            'Specify that argument when providing a .model file as the tokenizer_dir. '
            'It allows AutoTokenizer to instantiate the correct tokenizer type.')
        parser.add_argument('--vocab_file',
                            help="Used for sentencepiece tokenizers")
        parser.add_argument('--num_beams',
                            type=int,
                            help="Use beam search if num_beams >1",
                            default=1)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--top_k', type=int, default=1)
        parser.add_argument('--top_p', type=float, default=0.0)
        parser.add_argument('--length_penalty', type=float, default=1.0)
        parser.add_argument('--repetition_penalty', type=float, default=1.0)
        parser.add_argument('--presence_penalty', type=float, default=0.0)
        parser.add_argument('--frequency_penalty', type=float, default=0.0)
        parser.add_argument('--no_add_special_tokens',
                            dest='add_special_tokens',
                            default=True,
                            action='store_false',
                            help="Whether or not to add special tokens")
        parser.add_argument('--streaming', default=False, action='store_true')
        parser.add_argument('--streaming_interval',
                            type=int,
                            help="How often to return tokens when streaming.",
                            default=5)
        parser.add_argument(
            '--prompt_table_path',
            type=str,
            help="Path to .npy file, exported by nemo_prompt_convert.py")
        parser.add_argument(
            '--prompt_tasks',
            help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
        parser.add_argument('--lora_dir',
                            type=str,
                            default=None,
                            nargs="+",
                            help="The directory of LoRA weights")
        parser.add_argument(
            '--lora_task_uids',
            type=str,
            default=None,
            nargs="+",
            help="The list of LoRA task uids; use -1 to disable the LoRA module")
        parser.add_argument('--lora_ckpt_source',
                            type=str,
                            default="hf",
                            choices=["hf", "nemo"],
                            help="The source of lora checkpoint.")
        parser.add_argument(
            '--num_prepend_vtokens',
            nargs="+",
            type=int,
            help="Number of (default) virtual tokens to prepend to each sentence."
            " For example, '--num_prepend_vtokens=10' will prepend the tokens"
            " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
        parser.add_argument(
            '--medusa_choices',
            type=str,
            default=None,
            help="Medusa choice to use, if not none, will use Medusa decoding."
            "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
        )
        config = parser.parse_args()
        if isinstance(update_dict, dict):
            for k, v in update_dict.items():
                setattr(config, k, v)
        
        return config