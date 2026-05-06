from infill_workflow.utils import display
from infill_workflow.llm.model_list import *
import re

class ChatSession:
    """
    A generic class to manage chat sessions with different language models.
    This class can be used as a base class for specific implementations for
    different LLMs, including open-source models and API-only models.
    """

    def __init__(self, config, model_name):
        """
        Initializes the chat session with a configuration.

        :param config: A dictionary containing configuration settings.
        """

        if config.model_params.get('use_sampling_params', False):
            self.num_output_tokens = config.model_params.get('num_output_tokens', 1024)
            self.temperature = config.model_params.get('temperature', 1)
            self.top_k = config.model_params.get('top_k', 40)
            self.top_p = config.model_params.get('top_p', .9)
        else:
            self.num_output_tokens = 1024
            self.temperature = 1
            self.top_k = 40
            self.top_p = .9
        self.max_length = config.model_params.get('max_length', 4096)
            
        self.config = config
        self.msg_history = []
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }

        # set model name
        self.model_name = model_name

        # set whether to use the quantized version of a given LLM
        models_8bit = config.model_params.get('8bit_models', [])
        self.use_8bit = self.model_name in models_8bit

        models_4bit = config.model_params.get('4bit_models', [])
        self.use_4bit = self.model_name in models_4bit
        
    def get_response(self, user_message, system_message=None):
        """
        Retrieves a response from the language model.
        This method should be overridden in subclasses.

        :param message: The message to be sent to the model.
        :return: The response from the model.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def update_history(self, role, content):
        """
        Updates the message history.

        :param role: The role of the message sender ('user' or 'assistant').
        :param content: The content of the message.
        """
        self.msg_history.append({
            'role': role,
            'content': content
        })

    def get_history(self):
        """
        Returns the message history.

        :return: A list of message dictionaries.
        """
        return self.msg_history

    def get_usage(self):
        """
        Returns the usage statistics of the session.

        :return: A dictionary containing usage statistics.
        """
        return self.usage

    def __call__(self, message, system_role):
        """
        Shortcut for get_response.

        :param message: The message to be sent to the model.
        :return: The response from the model.
        """
        return self.get_response(message, system_role)

    def __str__(self):
        """
        Returns the message history as a JSON formatted string.
        """
        import json
        return json.dumps(self.msg_history, indent=4)

    def reset(self):
        pass

    def _extract_response(self,
                          prompts:   list | str,
                          responses: list | str):

        return_str = False
        if type(responses) == str:
            responses = [responses]
            return_str = True

        if type(prompts) == list and type(responses) == list:
            response = [resp[len(prompt):] for prompt, resp in zip(prompts, responses)]
        else:
            display.error('type mismatch between "prompts" and "responses". '
                          + f'prompts is type: {type(prompts)}, '
                          + f'responses is type: {type(responses)}')
            raise ValueError()

        if return_str:
            return response[0]
        else:
            return response
    def _prepare_openai_batch(self, usr_msg, sys_msg=None):
        # convert string input to list
        return_str=False
        if type(usr_msg) == str:
            usr_msg = [usr_msg]
            return_str=True

        if sys_msg is None:
            sys_msg = [None]*len(usr_msg)

        # ensure length of usr_msg and sys_msg match
        if len(usr_msg) != len(sys_msg):
            display.error('length of usr_msg does not match length of sys_msg')
            raise ValueError()
        batch = []
        for usr, sys in zip(usr_msg, sys_msg):
            if sys is not None:
                batch.append([
                    {'role': 'system', 'content': sys},
                    {'role': 'user',   'content': usr},
                ])
            else:
                batch.append([{'role': 'user', 'content': usr}])
        return batch, return_str
                
    
    def _prepare_batch(self, usr_msg, sys_msg=None):

        # convert string input to list
        return_str=False
        if type(usr_msg) == str:
            usr_msg = [usr_msg]
            return_str=True

            #return usr_msg, return_str


        if sys_msg is None:
            sys_msg = [None]*len(usr_msg)
        elif type(sys_msg) == str:
            sys_msg = [sys_msg]

        # ensure length of usr_msg and sys_msg match
        if len(usr_msg) != len(sys_msg):
            display.error('length of usr_msg does not match length of sys_msg')
            raise ValueError()

        # special case for palm and openai models
        if self.model_name in get_palm_models() \
                            + get_gpt_models() \
                            + get_gemini_models():
            return usr_msg, sys_msg, return_str

        msg = [self._preprocess_msg(prompt, sys) 
               for prompt, sys in zip(usr_msg, sys_msg)]
        
        return msg, return_str

    def _preprocess_msg(self, usr_msg, sys_msg=None):
        if self.model_name == 'Phind/Phind-CodeLlama-34B-v2':
            msg = f'### User Message\n{usr_msg}\n\n### Assistant\n'
            if sys_msg is not None:
                msg = f'### System Prompt\n{sys_msg}\n\n' + msg
            return msg
        elif self.model_name in get_dolly_instruction_models():
            msg = f'### Instruction:\n{usr_msg.strip()}\n\n### Response:'
            if sys_msg is not None:
                msg = f'{sys_msg.strip()}\n\n' + msg
            else:
                msg = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n' + msg
            return msg

        elif self.model_name in get_llama_chat_models():
            msg = f'[INST]{usr_msg.strip()}[/INST]'
            if sys_msg is not None:
                msg = f'<<SYS>>{sys_msg.strip()}<</SYS>>' + msg
            return msg

        elif 'Salesforce' in self.model_name and 'instruct' in self.model_name:
            msg = f'### Instruction:\n{usr_msg.strip()}\n\n### Response:\n'
            if sys_msg is not None:
                msg = f'{sys_msg.strip()}\n\n' + msg
            else:
                msg = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n' + msg

            return msg

        elif self.model_name == 'mistralai/Mistral-7B-Instruct-v0.1':
            msg = f'[INST] {usr_msg} [/INST]'
            return msg 

        elif 'lmsys/vicuna' in self.model_name \
          or 'falcon-7b-instruct' in self.model_name:
            msg = f'USER: {usr_msg.strip()}\nASSISTANT: '
            if sys_msg is not None:
                msg = sys_msg.strip() + '\n\n' + msg
            return msg
        elif re.search(r'Phi-3.*instruct', self.model_name):
            sys_msg = f'{sys_msg}\n\n' or ''
            msg = f'<|user|>\n{sys_msg}{usr_msg.strip()}<|end|>\n<|assistant|>'
            return msg
        else:
            breakpoint()
            return usr_msg
