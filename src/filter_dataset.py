#!/usr/bin/env python3
import json
import sys
import os
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Dict
from loguru import logger

# Template definition (copied from core/template.py)
@dataclass
class Template:
    template_name: str
    system_format: str
    user_format: str
    assistant_format: str
    tool_format: str
    function_format: str
    observation_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(
    template_name,
    system_format,
    user_format,
    assistant_format,
    tool_format,
    function_format,
    observation_format,
    system,
    stop_word=None,
):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        tool_format=tool_format,
        function_format=function_format,
        observation_format=observation_format,
        system=system,
        stop_word=stop_word,
    )

# Register qwen1.5 template
register_template(
    template_name="qwen1.5",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    tool_format="{content}",
    function_format="{content}",
    observation_format="<|im_start|>tool\n{content}<im_end>\n<|im_start|>assistant\n",
    system="You are a helpful assistant.",
    stop_word="<|im_end|>",
)

def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """Load tokenizer with same logic as validate.py"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if "gemma" in model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
        )

    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f"vocab_size of tokenizer: {tokenizer.vocab_size}")
    return tokenizer

def calculate_token_length(data, tokenizer, template, max_seq_length):
    """Calculate token length using same logic as UnifiedSFTDataset.__getitem__"""
    input_ids, target_mask = [], []

    # setting system information
    if template.system_format is not None:
        system = data["system"].strip() if "system" in data.keys() else template.system

        if system is not None:
            system_text = template.system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

    # setting tool information (if exists)
    if "tools" in data.keys() and data["tools"]:
        # Skip tools for now as they require additional processing
        pass

    conversations = data["conversations"]

    input_buffer = ""
    for conversation in conversations:
        role = conversation["role"]
        content = conversation["content"].strip()
        if role != "assistant":
            if role == "user":
                human = template.user_format.format(
                    content=content, stop_token=tokenizer.eos_token
                )
                input_buffer += human

            elif role == "function_call":
                # Skip function calls for now
                pass

            elif role == "observation":
                observation = template.observation_format.format(content=content)
                input_buffer += observation
        else:
            assistant = template.assistant_format.format(
                content=content, stop_token=tokenizer.eos_token
            )

            input_tokens = tokenizer.encode(
                input_buffer, add_special_tokens=False
            )
            output_tokens = tokenizer.encode(
                assistant, add_special_tokens=False
            )

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
            input_buffer = ""
    
    return len(input_ids)

def filter_dataset(input_file, output_file, model_name_or_path, template_name, max_seq_length):
    """Filter dataset to keep only samples with token length < max_seq_length"""
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {model_name_or_path}")
    tokenizer = load_tokenizer(model_name_or_path)
    
    # Get template
    if template_name not in template_dict.keys():
        raise ValueError(f"template_name doesn't exist, available: {template_dict.keys()}")
    template = template_dict[template_name]
    logger.info(f'Using template "{template_name}" for processing')
    
    # Load and process data
    logger.info(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data_lines = f.readlines()
    
    logger.info(f"Total samples: {len(data_lines)}")
    
    filtered_data = []
    truncated_count = 0
    
    for idx, line in enumerate(data_lines):
        data = json.loads(line.strip())
        
        try:
            token_length = calculate_token_length(data, tokenizer, template, max_seq_length)
            
            if token_length <= max_seq_length:
                filtered_data.append(line.strip())
            else:
                truncated_count += 1
                logger.debug(f"Sample {idx} has token length {token_length} > {max_seq_length}, excluded")
                
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Save filtered data
    logger.info(f"Saving filtered data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in filtered_data:
            f.write(line + '\n')
    
    logger.info(f"Filtering complete!")
    logger.info(f"Original samples: {len(data_lines)}")
    logger.info(f"Filtered samples: {len(filtered_data)}")
    logger.info(f"Excluded samples: {truncated_count}")
    logger.info(f"Retention rate: {len(filtered_data)/len(data_lines)*100:.2f}%")

if __name__ == "__main__":
    # Configuration matching test.sh
    input_file = "/workspace/llm-demo/data_characterX_v2/datasets/character_roleplay_pure_ko_en_20250729_082956.jsonl"
    output_file = "/workspace/llm-demo/data_characterX_v2/datasets/character_roleplay_pure_ko_en_20250729_082956_filtered.jsonl"
    model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
    template_name = "qwen1.5"
    max_seq_length = 4096
    
    logger.info(f"Starting dataset filtering...")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Template: {template_name}")
    logger.info(f"Max sequence length: {max_seq_length}")
    
    try:
        filter_dataset(input_file, output_file, model_name_or_path, template_name, max_seq_length)
        logger.info("✅ Filtering completed successfully!")
    except Exception as e:
        logger.error(f"❌ Filtering failed: {e}")
        sys.exit(1)