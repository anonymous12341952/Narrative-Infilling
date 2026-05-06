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

class HFGenerateSession(ChatSession):

    def __init__(self, config, model_name):
        """
        Initializes the huggingface generation chat session. This differs from the PipelineSession class by calling model.generate() instead of pipeline() to run inference.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        super().__init__(config, model_name)  # Initialize the base class with the loaded configuration

        self.max_length = config.get('max_length', 4096)
        self.num_output_tokens = config.get('num_output_tokens', 512)
        temperature = config.get('temperature', .1)
        top_k = config.get('top_k', 40)
        top_p = config.get('top_p', .9)
        num_beams = config.get('num_beams', 1)
        self.batch_size=config.get('batch_size', None)

        # import here because i can't see a better way to set the cache directory
        #os.environ['TRANSFORMERS_CACHE'] = config['model_cache']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
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

    def get_response(self, 
                     user_message:   str | list, 
                     system_message: str | list=None):
        """
        Retrieves a response from the generation model.
        """
        #msg = self._preprocess_msg(user_message, system_message)


        msg, return_str = self._prepare_batch(user_message, system_message)

        msg = self._preprocess_msg(user_message, system_message)

        if 'bigcode' in self.model_name:
            input_ids = self.tokenizer.encode(msg, return_tensors="pt")
            if torch.cuda.is_available():
                input_ids = input_ids.to('cuda')
            out_size = min(len(input_ids[0]) + self.num_output_tokens, self.max_length)
            generation_output = self.model.generate(
                input_ids,
                max_length=out_size,
            )
            response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        elif 'stablelm' in self.model_name:
            inputs = self.tokenizer(msg, return_tensors="pt")
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.num_output_tokens,
                temperature=0.75,
                top_p=0.95,
                do_sample=True,
            )
            response = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        elif 'incoder' in self.model_name:
            input_ids = self.tokenizer(msg, return_tensors="pt").input_ids
            out_size = min(len(input_ids[0]) + self.num_output_tokens, self.max_length)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_p=0.95,
                    temperature=.2,
                    max_length=out_size
                )
            response = self.tokenizer.decode(
                output.flatten(),
                clean_up_tokenization_spaces=False
            )
        elif 'PolyCoder' in self.model_name:
            input_ids = self.tokenizer.encode(msg, return_tensors='pt')
            out_size = min(len(input_ids[0]) + self.num_output_tokens, self.max_length)
            result = self.model.generate(
                input_ids,
                max_length=out_size,
                num_beams=4,
                #num_return_sequences=4
            )
            response = self.tokenizer.decode(result[0])
        elif self.model_name == 'mistralai/Mistral-7B-Instruct-v0.1':
            #ids = self.tokenizer.encode(msg, return_tensors="pt")
            ids = self.tokenizer(msg, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                ids = ids.to('cuda')
            generated_ids = self.model.generate(
                **ids,
                max_new_tokens=self.num_output_tokens,
                do_sample=True,
            )
            decoded = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            if return_str:
                return decoded[0]
            else:
                return decoded
        else:
            raise ValueError()
                       



