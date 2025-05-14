import torch
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

import os 
import re
import json
import argparse 
from tqdm import tqdm

'''
CUDA_VISIBLE_DEVICES=1 python model_responses/opensource/trained_model.py \
--peft_model_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/llama-3.1/checkpoint-1250 \
--output_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/query_500_sft_dpo_llama_epoch2.json 
'''

'''
CUDA_VISIBLE_DEVICES=4 python model_responses/opensource/trained_model.py \
--peft_model_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/qwen-2.5/checkpoint-1250 \
--output_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/query_500_sft_dpo_qwen_epoch2.json 
'''

'''
CUDA_VISIBLE_DEVICES=7 python model_responses/opensource/trained_model.py \
--peft_model_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/mistral/checkpoint-1250 \
--output_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/query_500_sft_dpo_mistral_epoch2.json 
'''

def main():
    args = parse_args()

    df = pd.read_csv(args.data_path)

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
    model.eval()

    # === [4] tokenizer 로드 ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === [5] 인퍼런스 입력 ===
    print('총 데이터 개수: ', len(df))

    all_responses = []

    for i in tqdm(range(len(df)), desc="Generating responses"):

        query = df['query'][i]
        system_prompt = 'Please respond in fluent English using a natural, conversational paragraph style. Do not use bullet points or numbered lists. Each response should consist of exactly 10 full sentences.'

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.8
            )

        input_length = len(tokenizer.encode(text))
        trimmed_output = outputs[0][input_length:]
        response = tokenizer.decode(trimmed_output, skip_special_tokens=True)
        all_responses.append(response)
    
    # === [6] 결과 저장 ===
    df['response'] = all_responses

    if args.output_path.endswith('.csv'):
        df.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    
    elif args.output_path.endswith('.json'):
        df.to_json(args.output_path, orient='records', force_ascii=False, indent=4)
   
    print(f"✅ 전체 저장 완료: {args.output_path}")
 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peft_model_path', type=str, default='outputs/dpo-run-001')
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/query_500_fin.csv')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--device', type=str, default='')
    return parser.parse_args()

if __name__ == "__main__":
    main()