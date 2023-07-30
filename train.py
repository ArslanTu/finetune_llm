import logging
import logging.config
import os

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)

from configs import (MODEL_DICT, TOKENIZER_DICT, FinetuneArguments,
                     logger_config)
from utils import (find_target_modules, load_belle_open_source_500k,
                   load_model, load_tokenizer)

# use FileLogger to print and write to file
# use StreamLogger to print only
logging.config.dictConfig(logger_config)
logger = logging.getLogger("StreamLogger")

def main():
    args, training_args, remain_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses(return_remaining_strings=True)
    if remain_args:
        logger.warning(f"Excess args: {remain_args}")
        raise ValueError(f"Excess args: {remain_args}")

    set_seed(training_args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", None)) # maybe default 0
    logger.info(f"world size {world_size} local rank {local_rank}")

    ### prepare model ###
    model = load_model(
        model_name_or_path=args.model_name_or_path,
        model_cls=MODEL_DICT[args.model_name_or_path]['cls'],
        half=args.half,
        quantization=args.quantization,
        local_rank=local_rank,)
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer = load_tokenizer(
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_cls=TOKENIZER_DICT[tokenizer_name_or_path]['cls'],)

    logger.debug(f"prepare model for lora training")
    # TODO: find out when to use it
    model = prepare_model_for_kbit_training(model)
    target_modules = args.lora_target_modules
    if target_modules is not None:
        target_modules = target_modules.split('.')
        logger.debug(f"use the target_modules you set: {target_modules}")
    else:
        target_modules = find_target_modules(model, quantization=args.quantization)
        logger.debug(f"auto set target_modules to {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
        task_type=args.lora_task_type,
    )
    logger.debug(f"LoRA config: {lora_config}")
    model = get_peft_model(model, lora_config)  

    ### prepare data ###
    data = eval(args.data_func_name)(tokenizer, args.data_path)
    if args.train_size > 0:
        data = data.shuffle(seed=training_args.seed).select(range(args.train_size))

    if args.test_size > 0:
        train_val = data.train_test_split(
            test_size=args.test_size, shuffle=True, seed=training_args.seed
        )
        train_data = train_val["train"].shuffle(seed=training_args.seed)
        val_data = train_val["test"].shuffle(seed=training_args.seed)
    else:
        train_data = data['train'].shuffle(seed=training_args.seed)
        val_data = None

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer,
                                             pad_to_multiple_of=8,
                                             return_tensors="pt",
                                             padding=True)
    )
    logger.debug("Start training.")
    trainer.train(resume_from_checkpoint=False)
    logger.debug("Finish training, start saving model.")
    model.save_pretrained(training_args.output_dir)
    logger.debug("Finish saving, exit.")
if __name__ == "__main__":
    main()