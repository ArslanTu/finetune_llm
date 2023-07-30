from dataclasses import dataclass, field


@dataclass
class FinetuneArguments:
    ### basic ###
    # a path or identifier of pretrained model
    model_name_or_path: str = field() 
    # the function name to load dataset
    data_func_name: str = field() 
    data_path: str = field()
    # default -1 to use all data
    train_size: int = field(default=-1)
    test_size: int = field(default=200)
    # a path or identifier of pretrained tokenizer
    # if None, use the tokenizer of model_name_or_path
    tokenizer_name_or_path: str = field(default=None)
    # max token length
    max_len: int = field(default=2048)
    quantization: str = field(default=None)
    half: bool = field(default=False)
    
    ### for lora training ###
    # reference: https://github.com/HarderThenHarder/transformers_tasks/issues/28
    lora_rank: int = field(default=8)
    lora_alpha:  int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default='none')
    # target modules for LoRA, commonly set automatically
    lora_target_modules: str = field(default=None)
    lora_task_type: str = field(default='CAUSAL_LM')