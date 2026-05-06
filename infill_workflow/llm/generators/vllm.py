from typing import List, Tuple
from functools import wraps
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

from .base import Generator
from infill_workflow.llm.session import Session

class VLLM(Generator):
    
    def __init__(self,
        *args,
        **kwargs,
    ):
        super(VLLM, self).__init__(*args, **kwargs)

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        if kwargs.get('model_cache'):
            kwargs.pop('model_cache')
        self.model = LLM(
            model=self.model_name,
            seed=int(time.time()),
            download_dir=self.model_cache,
            enforce_eager=True,
            **kwargs,
        )

    @wraps(Generator.generate)
    def generate(self, 
        sessions: List[Session],
        prompts: List[str],
        temp: float=None,
        use_tqdm: bool=False,
    ) -> Tuple[List[Session], List[str]]:

        # prepare model input
        out_sessions = [
            sess.add_prompt(prompt)
            for sess, prompt in zip(sessions, prompts)
        ]

        sampling_params = SamplingParams(
            temperature=temp or self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            max_tokens=self.max_length,
        )

        if self.session_type == 'chat':
            chats = [sess.get_hist() for sess in out_sessions]
            # make inference call
            responses = self.model.chat(
                chats, sampling_params, use_tqdm=use_tqdm
            )
        # TODO
        elif self.session_type == 'chat_custom':
            raise NotImplementedError()
        # TODO
        elif self.session_type == 'standard':
            raise NotImplementedError()
        
        # convert responses to text
        responses = [resp.outputs[0].text for resp in responses]

        # add model responses to history
        out_sessions = [
            sess.add_prompt(resp, role='assistant')
            for sess, resp in zip(out_sessions, responses)
        ]

        return out_sessions, responses