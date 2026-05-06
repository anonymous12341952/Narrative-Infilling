import openai
import os
from openai import OpenAI
from tqdm import tqdm

# local imports
from .chat_session import ChatSession
from infill_workflow.utils import (
    display,
)

class OpenAISession(ChatSession):
    """
    A subclass of ChatSession specifically for OpenAI's language models, using YAML configuration.
    """

    def __init__(self, config, model_name):
        """
        Initializes the OpenAI chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used (e.g., '3.5' or '4').
        """
        # # Load configuration from YAML file
        # with open(config_path, 'r') as file:
        #     full_config = yaml.safe_load(file)
        
        # config = full_config['data_path']
        self.num_retries = 3
        
        super().__init__(config, model_name)  # Initialize the base class with the loaded configuration
        self.session= OpenAI()
        # Initialize usage statistics
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }

    def get_response(self,
                     user_message: list | str,
                     system_message: list | str=None,
                     prog_bar: bool=False):
        """
        Retrieves a response from OpenAI's language model.

        :param user_message: The user message to be sent to the model.
        :param system_message: An optional system message.
        :return: The response from the model.
        """
        responses = []
        messages, return_str = self._prepare_openai_batch(user_message, system_message)

        # decide whether to use progress bar
        if prog_bar:
            msg_iterator = tqdm(messages, total=len(messages))
        else:
            msg_iterator = messages

        for msg in msg_iterator:
            for try_num in range(self.num_retries):
                try:
                    response = self.session.chat.completions.create(
                        model=self.model_name,
                        messages=msg,
                    )
                    responses.append(response.choices[0].message.content)
                    break
                except KeyboardInterrupt as e:
                    print(e)
                    import sys
                    sys.exit()
                except Exception as e:
                    if try_num == 2:
                        display.error('try limit reached for OpenAI API call')
                        responses.append('TIMEOUT ERROR')
        if return_str:
            return responses[0]
        else:
            return responses
        

    """
    deprecated because i am rewriting it to support batching.
    however, i want to bring back history updating at some point so i am not totally
    getting rid of this
    """
    def _get_response(self,
                     user_message: list | str,
                     system_message: list | str=None,):
        """
        Retrieves a response from OpenAI's language model.

        :param user_message: The user message to be sent to the model.
        :param system_message: An optional system message.
        :return: The response from the model.
        """
        
        if system_message:
            self.update_history('system', system_message)
            self.usage['prompt_tokens'] += len(system_message.split())

        if isinstance(user_message, str):
            user_message = [user_message]

        responses = []
        for msg in user_message:
            self.update_history('user', msg)

            try:
                client = OpenAI()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.msg_history,
                )
            except KeyboardInterrupt as e:
                print(e)
                import sys; sys.exit(0)
            except openai.OpenAIError as e:
                display.error(f'Error occurred during Openai API call.\n\tError message: {e}')
                return None
            except Exception as e:
                display.error(f"An error unrelated to the OpenAI API occurred: {e}")
                return None

            response_content = response.choices[0].message.content
            responses.append(response_content)
            self.update_history('assistant', response_content)

            # Update usage statistics
            self.usage['completion_tokens'] += len(response_content.split())
            self.usage['prompt_tokens'] += len(msg.split())
            self.usage['total_tokens'] = self.usage['completion_tokens'] + self.usage['prompt_tokens']

        return responses if len(responses) > 1 else responses[0]
