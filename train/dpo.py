import os 
import re
import json
import argparse 

from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
CUDA_VISIBLE_DEVICES=1,3 accelerate launch \
  --multi_gpu \
  train/dpo.py \
  --input_path data/train.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_path ./dpo_output
  --hug_token huggingface_token
"""

def main():
    args = parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    train_dataset = Dataset.from_list(raw_data)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token = args.hug_token,
        trust_remote_code=True 
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = DPOConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3, 
        num_train_epochs=5,
        # evaluation_strategy="no",
        report_to=args.wandb_run_name
        )
    
    os.environ["WANDB_PROJECT"] = args.wandb_project

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
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/preference_data.jsonl')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/outputs/safe_dpo/llama-3.1-8b-instruct')
    parser.add_argument('--wandb_project', type=str, default="AHa-DPO")
    parser.add_argument('--wandb_run_name', type=str, default="trial1")
    parser.add_argument('--hug_token', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    main()