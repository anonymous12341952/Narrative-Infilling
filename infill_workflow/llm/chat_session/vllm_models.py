#external imports
import os
import yaml
import torch
import transformers
import time
from vllm import LLM, SamplingParams
from vllm.transformers_utils.config import get_config
import ray
import sys

# import config classes for checking number of attention heads
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import (
    MistralConfig
)


# local imports
from .chat_session import ChatSession
from infill_workflow.utils import display

class VLlmSession(ChatSession):
    """
    A subclass of ChatSession specifically for the LLAMA2 language model, using YAML configuration.
    """

    def __init__(self, config, model_name):
        """
        Initializes the LLAMA2 chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        
        super().__init__(config, model_name)  # Initialize the base class with the loaded configuration

        # Set up LLAMA2 API credentials and model
        # Assuming LLAMA2 uses an API key and has a similar setup to OpenAI

        model_cache = config.model_params.get('model_cache', None)

        # set number of gpus to use
        num_devices = config.model_params.get('num_devices', 1)
        tensor_parallel_size = self._set_tensor_parallel(num_devices)

        # Initialize usage statistics
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.num_output_tokens,
        )

        if config.model_params['dtype'] == 'float16':
            dtype = torch.float16
        elif config.model_params['dtype'] == 'auto':
            dtype = 'auto'

        self.model = LLM(
            model_name,
            trust_remote_code=True,
            download_dir=model_cache,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            seed=int(time.time()),
            max_model_len=self.max_length,
            enforce_eager=True,
            worker_use_ray=True,
        )

    def get_response(
                self,
                user_message: str | list,
                system_message: str | list=None,
                prog_bar=True):
        """
        Retrieves a response from the vLLM language model.
        """

        msg, return_str = self._prepare_batch(
            user_message, 
            system_message
        )
        # Implement the logic to interact with the LLAMA2 model's API
        # This is a placeholder implementation

        # generate response
        seqs = self.model.generate(
            msg,
            sampling_params=self.sampling_params,
            use_tqdm=prog_bar
        )


        # Update history and usage statistics
        # [Rest of the method should handle response parsing and 
        # updating the session similar to OpenAISession]
        if return_str:
            response = seqs[0].outputs[0].text
        else:
            response = [seq.outputs[0].text for seq in seqs]

        return response


    def _set_tensor_parallel(self, num_devices):

        # get number of attention heads for the model
        n_head = self._get_num_attn_heads()

        tensor_parallel_size = num_devices
        while n_head%tensor_parallel_size != 0:
            tensor_parallel_size -= 1

        return tensor_parallel_size

    def _get_num_attn_heads(self):
        
        llm_cfg = get_config(self.model_name, trust_remote_code=True)
        try:
            if (isinstance(llm_cfg, LlamaConfig)
                    or isinstance(llm_cfg, MistralConfig)
                    or 'Phi3Config' in str(type(llm_cfg))):
                n_head = llm_cfg.num_attention_heads
            elif 'mosaicml' in self.model_name:
                n_head = llm_cfg.n_heads
            else:
                n_head = llm_cfg.n_head
        except KeyboardInterrupt:
            sys.exit()
        except:
            display.warning(
                'search through `llm_cfg` to find the argument for # '
                'of attention heads'
            )
            breakpoint()


        return n_head
