import os 
import re
import json
import argparse 
import wandb

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

"""
CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
--multi_gpu \
--main_process_port 1235 \
train/dpo.py \
--base_model /home/data3/users/jiwon/outputs/safe_real_fin/sft/llama-3.1/checkpoint-471 \
--output_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/llama-3.1 \
--wandb_run_name sfa-dpo-llama
"""

"""
CUDA_VISIBLE_DEVICES=3 python train/dpo.py \
--peft_model_path /home/data3/users/jiwon/outputs/safe_real_fin/sft/mistral/checkpoint-471 \
--output_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/mistral \
--wandb_run_name sft-dpo-mistral
"""

def main():
    args = parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    train_dataset = Dataset.from_list(raw_data)

    # === [1] LoRA 설정 불러오기 ===
    peft_config = PeftConfig.from_pretrained(args.peft_model_path)
    base_model_path = peft_config.base_model_name_or_path

    # === [2] base model 로드 ===
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"  
    )

    # === [3] LoRA 적용 모델 불러오기 ===
    model = PeftModel.from_pretrained(
        base_model, 
        args.peft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"  
        )
    
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        padding_side="right", 
        use_fast=False
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = DPOConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3, 
        num_train_epochs=3,
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
    parser.add_argument('--peft_model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking_fin/preference_data.json')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--wandb_project', type=str, default="AHa-DPO")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--hug_token', type=str, default='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')
    return parser.parse_args()

if __name__ == "__main__":
    main()