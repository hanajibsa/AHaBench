import os 
import re
import json
import argparse 
import wandb

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

"""
CUDA_VISIBLE_DEVICES=1,3 accelerate launch \
  --multi_gpu \
  train/dpo.py \
  --input_path data/train.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_path ./dpo_output
  --wandb_run_name trial1
  --hug_token huggingface_token
"""

def main():
    args = parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    train_dataset = Dataset.from_list(raw_data)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token = args.hug_token,
        trust_remote_code = True,
        torch_dtype = torch.bfloat16,
        # attn_implementation="flash_attention_2"
        )
    
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        ]
    )
    model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        padding_side="right", 
        use_fast=False
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = DPOConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3, 
        num_train_epochs=5,
        # evaluation_strategy="no",
        report_to="wandb",
        run_name=args.wandb_run_name,
        remove_unused_columns=False
        )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/preference_data.json')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/outputs/safe_dpo/llama-3.1-8b-instruct')
    parser.add_argument('--wandb_project', type=str, default="AHa-DPO")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--hug_token', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    main()