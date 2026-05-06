import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tqdm import tqdm
import time
import sys

# local imports
from .chat_session import ChatSession
from infill_workflow.utils import (
    display,
)

class GeminiSession(ChatSession):
    """
    A subclass of ChatSession specifically for Google's language models,
    using YAML configuration.
    """

    def __init__(self, config, model_name):
        """
        Initializes the OpenAI chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used 
            (e.g., '3.5' or '4').
        """

        # Initialize the base class with the loaded configuration 
        super().__init__(config, model_name) 
        genai.configure(api_key=config.api_keys['palm'])
        self.model = genai.GenerativeModel(model_name)

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list=None,
                     prog_bar: bool=False):
        """
        Retrieves a response from a PALM language model.

        :param user_message: The user message to be sent to the model.
        :param system_message: An optional system message.
        :return: The response from the model.
        """

        usr_msg, sys_msg, return_str = self._prepare_batch(
            user_message, 
            system_message
        )

        # decide whether to use progress bar
        if prog_bar:
            msg_iterator = tqdm(
                zip(usr_msg, sys_msg), 
                total=len(usr_msg)
            )
        else:
            msg_iterator = zip(usr_msg, sys_msg)

        responses = []
        start = time.time()
        num_queries = 0
        #for prompt, context in tqdm(zip(usr_msg, sys_msg), total=len(usr_msg)):
        for idx, (prompt, context) in enumerate(msg_iterator):
            done=False
            num_attempts=0
            while not done:
                num_attempts+=1
                try:
                    chat = self.model.start_chat(history=[])
                    response = chat.send_message(
                        content=prompt,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: \
                                HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: \
                                HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HARASSMENT: \
                                HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: \
                                HarmBlockThreshold.BLOCK_NONE},
                    )
                    done=True
                except KeyboardInterrupt:
                    display.info('Keyboard interrupt occurred. Exiting ...')
                    sys.exit(1)
                except Exception as e:
                    display.warning(f'Exception message: {e}')
                    display.warning(
                        f'Exception occured for sample {idx}. '
                        f'Number of attempts: {num_attempts}'
                    )
                    #sys.exit(1)
            #reply = session.get_response(msg)
            responses.append(response.text)
            num_queries += 1
            end = time.time()

            # set cooldown time due query limits (https://ai.google.dev/pricing)
            """
            if num_queries >= 60 and end-start > 55:
                time.sleep(int(65-(end-start)))
                start = time.time()
                num_queries = 0
            """

        if return_str:
            return responses[0]
        else:
            return responses
