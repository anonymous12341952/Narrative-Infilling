from .chat_session import ChatSession
from .openai_gpt import OpenAISession
from .hf_pipeline_models import PipelineSession
from .hf_generate_models import HFGenerateSession
from .wizardcoder import WizardCoderSession
from .salesforce import SalesforceSession
from .vllm_models import VLlmSession
from .gemini import GeminiSession

from infill_workflow.utils import (
    display,
)
from infill_workflow.llm.model_list import *

def select_chat_model(cfg: dict, model_name: str) -> ChatSession:
    """
    returns a ChatSession object given the model name and config
    Input:
        cfg[dict]: the config to initialize the object
        model_name[str]: name of the model

    Output:
        ChatSession object
    """

    if model_name in get_gpt_models():
        return  OpenAISession(cfg, model_name)
    elif model_name in get_gemini_models():
        return GeminiSession(cfg, model_name)
    elif model_name in get_vllm_models():
        return VLlmSession(cfg, model_name)
    elif model_name in get_pipeline_models():
        return PipelineSession(cfg, model_name)
    elif model_name in get_wizardlm_models():
        return WizardCoderSession(cfg, model_name)
    elif model_name in get_salesforce_models():
        return SalesforceSession(cfg, model_name)
    elif model_name in get_hf_generate_models():
        return HFGenerateSession(cfg, model_name)
    else:
        display.error(f'model: {model_name} is an unsupported option')
        raise ValueError()
