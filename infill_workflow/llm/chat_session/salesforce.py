#external imports
import os
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# local imports
from .chat_session import ChatSession
from infill_workflow.utils import display

class SalesforceSession(ChatSession):

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
        self.temperature = config.get('temperature', .1)

        # import here because i can't see a better way to set the cache directory
        #os.environ['TRANSFORMERS_CACHE'] = config['model_cache']

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if 'codet5' in model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                torch_dtype=torch.float16,
                #low_cpu_mem_usage=True,
                device_map='auto',
                trust_remote_code=True,
                cache_dir=config['model_cache'],
                load_in_8bit=True
            )

        elif 'codegen' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
                cache_dir=config['model_cache'],
            )
        else:
            display.error(f'model \'{model_name}\' is not a supported model for the SalesforceSession class')
            raise ValueError()

    def get_response(self, user_message, system_message=None):
        """
        Retrieves a response from the generation model.
        """
        msg = self._preprocess_msg(user_message, system_message)
        if 'codet5' in self.model_name:
            encoding = self.tokenizer(msg, return_tensors="pt")
            encoding['decoder_input_ids'] = encoding['input_ids'].clone()
            out_size = min(
                len(encoding['decoder_input_ids'][0]) + self.num_output_tokens, 
                self.max_length
            )
            outputs = self.model.generate(**encoding, max_length=out_size)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif 'codegen' in self.model_name:
            input_ids = self.tokenizer(msg, return_tensors="pt").input_ids
            out_size = min(len(input_ids[0]) + self.num_output_tokens, self.max_length)
            generated_ids = self.model.generate(input_ids, max_length=out_size)
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return response


