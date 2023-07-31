import logging
import os

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

logger = logging.getLogger("StreamLogger")

def load_model(
    model_name_or_path: str, 
    model_cls,
    lora_path: str=None, 
    half: bool=False,
    quantization=None, 
    local_rank=None, 
    device_map=None, 
    lora_device_map=None, 
    inference: bool=False,
    **kwargs):
    """
    load pretrained model and lora model

    Args:
        model_name_or_path (str): a path to a pretrained model or a model identifier from huggingface.co/models
        model_cls (_type_): the class used to load the model, should be a subclass of PreTrainedModel
        lora_path (str, optional): the path to lora weights. Defaults to None.
        half (bool, optional): whether to use half. Defaults to False.
        quantization (_type_, optional): quantization type, '8bit' or '4bit'. Defaults to None.
        local_rank (_type_, optional): local_rank, in distributed environment. Defaults to None.
        device_map (_type_, optional): base model device_map. Defaults to None.
        lora_device_map (_type_, optional): lora weights device_map. Defaults to None.
        inference (bool, optional): whether inference mode. Defaults to False.

    Returns:
        _type_: model
    """
    logger.debug(f"load model from {model_name_or_path}")
    # avoid distributed inference
    if local_rank is None:
        local_rank = os.environ.get('LOCAL_RANK', None)
    if local_rank is not None and inference == True:
        raise ValueError("currently don't support distributed inference")

    # avoid training arter load lora
    if lora_path is not None and inference == False:
        raise ValueError("currently donn't support training after load lora")

    # check quantization type
    assert (quantization is None or quantization in ['8bit', '4bit']), "quantization should be '8bit' or '4bit'"

    # set device_map
    # in a distributed training environment
    if local_rank is not None:
        # always use the gpu in local_rank
        default_device_map = {'': int(local_rank)}
        if device_map is None:
            logger.debug(f"auto set device_map to {default_device_map}")
        else:
            logger.debug(f"abandon the device_map you set, use {default_device_map} instead")
        device_map = default_device_map
    # not in a distributed training environment, use auto or you set
    else:
        if device_map is None:
            device_map = 'auto'
            logger.debug(f"auto set device_map to {device_map}")
        else:
            logger.debug(f"use the device_map you set: {device_map}")

    # set lora_device_map
    if lora_device_map is None:
        lora_device_map = 'auto'
        logger.debug(f"auto set lora_device_map to {lora_device_map}")
    else:
        logger.debug(f"use the lora_device_map you set: {lora_device_map}")

    # set model_dtype
    model_dtype = torch.float32
    # if half:
    #     if inference or quantization is not None:
    #         model_dtype =  torch.float16
    #     else:
    #         model_dtype = torch.bfloat16
    if half:
        model_dtype = torch.float16
    logger.debug(f"set model_dtype to {model_dtype}")
    
    # set quantization_config
    quantization_config = None
    if quantization == '8bit':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, 
            llm_int8_threshold=6.0)
    elif quantization == '4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4",)
    logger.debug(f"set quantization_config to {quantization_config}")


    # load model
    model = model_cls.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=model_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if lora_path is not None:
        logger.debug(f"load lora from {lora_path}")
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map=lora_device_map,
        )
    # FIXME: half after load in fp16?
    # if half and quantization is not None:
    #     logger.debug(f"convert model to half precision")
    #     model.half()

    if inference:
        logger.debug(f"set model to eval mode")
        model.eval()
    
    return model

def load_tokenizer(
        tokenizer_name_or_path: str,  
        tokenizer_cls,
        **kwargs):
    """
    load tokenizer from pretrained or cache

    Args:
        tokenizer_name_or_path (str): a name or path to a pretrained tokenizer
        tokenizer_cls (_type_): the class used to load the tokenizer, should be a subclass of PreTrainedTokenizer

    Returns:
        _type_: tokenizer
    """
    logger.debug(f"load tokenizer from {tokenizer_name_or_path}")
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return tokenizer
