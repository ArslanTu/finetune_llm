import logging

from datasets import load_dataset

logger = logging.getLogger("StreamLogger")

def load_belle_open_source_500k(
        tokenizer, 
        data_path: str, 
        max_len=2048, 
        num_proc: int=1):
    """
    Load the Belle open source 500k dataset, and tokenize it with the given tokenizer.
    You can download data file with: 
    https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/blob/main/Belle_open_source_0.5M.json
    
    Args:
        tokenizer (tokenizer): transformers tokenizer
        data_path (str, optional): the path to data file. Defaults to './data/Belle_open_source_0.5M.json'.
        max_len (int, optional): max token length. Defaults to 2048.
        num_proc (int, optional): number of processes to map data. Defaults to 1.

    Returns:
        DataSet: Dataset with tokenized data
    """
    logger.debug(f"Load belle_open_source_500k, max_len: {max_len}, num_proc: {num_proc}")
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < max_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= max_len:
            result["input_ids"][max_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][max_len - 1] = 1

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        instruction = data_point['instruction']
        input_text = data_point["input"]
        input_text = "<|User|>:" + instruction + input_text + "<eoh>\n<|Bot|>:"
        target_text = data_point["output"] + "<eoa>\n"
        full_prompt = input_text + target_text
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_path)["train"]
    data = data.map(generate_and_tokenize_prompt, num_proc=num_proc)
    return data