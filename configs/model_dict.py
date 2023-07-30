from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DICT = {
    "internlm/internlm-chat-7b": {
        "cls": AutoModelForCausalLM,
    }
}

TOKENIZER_DICT = {
    "internlm/internlm-chat-7b": {
        "cls": AutoTokenizer,
    }
}