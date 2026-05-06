#external imports
import os
import yaml
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# local imports
from .chat_session import ChatSession

class WizardCoderSession(ChatSession):

    def __init__(self, config, model_name):
        """
        Initializes the huggingface generation chat session. This differs from the PipelineSession class by calling model.generate() instead of pipeline() to run inference.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        super().__init__(config, model_name)  # Initialize the base class with the loaded configuration

        self.model_name = model_name  # Model name is directly used
        self.max_length = config.get('max_length', 4096)
        self.num_output_tokens = config.get('num_output_tokens', 512)
        temperature = config.get('temperature', .1)
        top_k = config.get('top_k', 40)
        top_p = config.get('top_p', .9)
        num_beams = config.get('num_beams', 1)

        # import here because i can't see a better way to set the cache directory
        #os.environ['TRANSFORMERS_CACHE'] = config['model_cache']

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            #torch_dtype=torch.float16,
            torch_dtype=torch.float32,
            device_map='auto',
            trust_remote_code=True,
            cache_dir=config['model_cache'],
        )

        self.generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def get_response(self, user_message, system_message=None):
        """
        Retrieves a response from the generation model.
        """
        msg, return_str = self._prepare_batch(user_message, system_message)
        input_ids = self.tokenizer(
            msg, 
            padding=True,
            return_tensors="pt"
        )#.input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        out_size = min(len(input_ids[0]) + self.num_output_tokens, self.max_length)
        with torch.no_grad():
            generation_output = self.model.generate(
                **input_ids,
                max_length=out_size,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
            )
        seq = generation_output.sequences
        if return_str:
            response = self.tokenizer.decode(seq[0], skip_special_tokens=True)
        else:
            response = self.tokenizer.batch_decode(seq, skip_special_tokens=True)

        response = self._extract_response(msg, response)
        return response


