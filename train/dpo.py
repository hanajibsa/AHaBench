import os 
import re
import json
import argparse 

# from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    args = parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        train_dataset = json.load(f)
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    training_args = DPOConfig(output_dir=args.output_path, logging_steps=10)
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/ranking_result.json')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/outputs/safe_dpo/llama-3.1-8b-instruct')
    return parser.parse_args()

if __name__ == "__main__":
    main()