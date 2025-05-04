import os 
import re
import json
import argparse 
import pandas as pd

model_response_keys = {
    "qwen_response": "qwen_responses",
    "llama_response": "llama_responses",
    "mistral_response": "mistral_responses",
    "gpt3.5_response": "gpt3.5_responses",
    "gpt4o_response": "gpt4o_responses"
}

def main():
    args = parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dpo_samples = []

    for record in data:
        query = record['rephrased_query']
        sorted_scores = record["sorted_model_scores"]

        best_model, best_score = next(iter(sorted_scores.items()))
        worst_model, worst_score = next(reversed(sorted_scores.items()))

        chosen = record[model_response_keys[best_model]]
        rejected = record[model_response_keys[worst_model]]

        dpo_samples.append({
            "prompt": query,
            "chosen": chosen,
            "rejected": rejected,
            "score_chosen": best_score,
            "score_rejected": worst_score
        })

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 전체 저장 완료: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/ranking_result.json')
    parser.add_argument('--output_path', type=str, default='')
    return parser.parse_args()

if __name__ == "__main__":
    main()