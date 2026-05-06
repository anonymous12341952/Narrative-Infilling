
def get_all_models() -> list:
    return sorted(
        get_instruct_models()
        + get_generation_models()
    )

def get_gpt_models() -> list:
    return [
        'gpt-3.5-turbo',
        'gpt-4',
        # 'gpt-4-32',
    ]

def get_palm_models() -> list:
    return ['chat-bison-001',]

def get_gemini_models() -> list:
    return ['gemini-pro',]

def get_instruct_models() -> list:
    return sorted([
        'Phind/Phind-CodeLlama-34B-v2',
        'WizardLM/WizardCoder-1B-V1.0',
        'WizardLM/WizardCoder-3B-V1.0',
        'WizardLM/WizardCoder-15B-V1.0',
        'codellama/CodeLlama-7b-Instruct-hf',
        'codellama/CodeLlama-13b-Instruct-hf',
        'codellama/CodeLlama-34b-Instruct-hf',
        'Salesforce/codegen25-7b-instruct',
        'Salesforce/instructcodet5p-16b',
        'mistralai/Mistral-7B-Instruct-v0.1',
        "mistralai/Mistral-7B-Instruct-v0.2"
        'lmsys/vicuna-7b-v1.5',
        'lmsys/vicuna-7b-v1.5-16k',
        'lmsys/vicuna-13b-v1.5',
        'lmsys/vicuna-13b-v1.5-16k',
    ])

def get_pipeline_models() -> list:
    return []

def get_wizardlm_models() -> list:
    models = get_instruct_models()
    models = list(filter(lambda x: 'WizardLM' in x, models))

    return models

def get_llama_chat_models() -> list:
    return [
        'Phind/Phind-CodeLlama-34B-v2',
        'codellama/CodeLlama-7b-Instruct-hf',
        'codellama/CodeLlama-13b-Instruct-hf',
        'codellama/CodeLlama-34b-Instruct-hf',
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Llama-2-70b-chat-hf',
        'meta-llama/Meta-Llama-3-8B-Instruct',
    ]
def get_dolly_instruction_models() -> list:
    return [
        'WizardLM/WizardCoder-1B-V1.0',
        'WizardLM/WizardCoder-3B-V1.0',
        'WizardLM/WizardCoder-15B-V1.0',
        'mosaicml/mpt-7b-instruct',
        'mosaicml/mpt-7b-chat',
        'mosaicml/mpt-30b-chat',
        'mosaicml/mpt-30b-instruct',
    ]

def get_salesforce_models() -> list:
    #models = get_instruct_models()
    models = get_all_models()
    models = list(filter(lambda x: 'Salesforce' in x, models))
    
    if 'Salesforce/codegen25-7b-instruct' not in models:
        models.append('Salesforce/codegen25-7b-instruct')

    return models

def get_vllm_models() -> list:
    return sorted([
        'mistralai/Mistral-7B-v0.1',
        'mistralai/Mistral-7B-Instruct-v0.1',
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        'Phind/Phind-CodeLlama-34B-v2',
        'codellama/CodeLlama-7b-Instruct-hf',
        'codellama/CodeLlama-13b-Instruct-hf',
        'codellama/CodeLlama-34b-Instruct-hf',
        'lmsys/vicuna-7b-v1.5',
        'lmsys/vicuna-7b-v1.5-16k',
        'lmsys/vicuna-13b-v1.5',
        'lmsys/vicuna-13b-v1.5-16k',
        'bigcode/starcoder',
        'bigcode/gpt_bigcode-santacoder',
        'bigcode/santacoder',
        'bigcode/starcoder',
        'bigcode/starcoderplus',
        'EleutherAI/gpt-j-6b',
        'EleutherAI/gpt-neo-2.7B',
        'EleutherAI/gpt-neox-20b',
        'tiiuae/falcon-7b-instruct',
        'tiiuae/falcon-40b-instruct',
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Llama-2-70b-chat-hf',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'mosaicml/mpt-7b-instruct',
        'mosaicml/mpt-7b-chat',
        'mosaicml/mpt-30b-chat',
        'mosaicml/mpt-30b-instruct',
        'microsoft/Phi-3-mini-4k-instruct',
        'microsoft/Phi-3-mini-128k-instruct',
    ])

def get_generation_models() -> list:
    return sorted([
        'bigcode/santacoder', # other versions exist
        'bigcode/starcoder',
        'bigcode/starcoderplus',
        'codellama/CodeLlama-7b-hf',
        'codellama/CodeLlama-13b-hf',
        'codellama/CodeLlama-34b-hf',
        'Salesforce/codegen-2B-multi',
        'Salesforce/codegen-2B-nl',
        'Salesforce/codegen-6B-multi',
        'Salesforce/codegen-6B-nl',
        'Salesforce/codegen-16B-multi',
        'Salesforce/codegen-16B-nl',
        'Salesforce/codegen2-1B',
        'Salesforce/codegen2-3_7B',
        'Salesforce/codegen2-7B',
        'Salesforce/codegen2-16B',
        'Salesforce/codet5p-2b', # not sure
        'Salesforce/codet5p-6b', # not sure
        'Salesforce/codet5p-16b', # not sure
        'mistralai/Mistral-7B-v0.1'
        'facebook/incoder-1B',
        'facebook/incoder-6B',
        'EleutherAI/gpt-j-6b',
        'EleutherAI/gpt-neo-2.7B',
        'EleutherAI/gpt-neox-20b',
        'NinedayWang/PolyCoder-2.7B',
        'stabilityai/stablelm-base-alpha-7b-v2', # not sure
    ])

def get_hf_generate_models() -> list:
    models = sorted([
        'bigcode/santacoder', # other versions exist
        'bigcode/starcoder',
        'bigcode/starcoderplus',
        'facebook/incoder-1B',
        'facebook/incoder-6B',
        'stabilityai/stablelm-base-alpha-7b-v2', # not sure
        'NinedayWang/PolyCoder-2.7B',
        'mistralai/Mistral-7B-Instruct-v0.1',
    ])
    return models
