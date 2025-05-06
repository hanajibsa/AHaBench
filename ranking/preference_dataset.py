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

    for i, record in enumerate(data):
        query = record['rephrased_query']
        sorted_scores = record["sorted_model_scores"]

        sorted_models = list(sorted_scores.items())  # [('gpt3.5_response', 16.5), ..., ('mistral_response', 13.0)]

        best_model, best_score = sorted_models[0]
        worst_model, worst_score = sorted_models[-1]

        chosen = record[model_response_keys[best_model]]
        rejected = record[model_response_keys[worst_model]]

        if rejected is None or (isinstance(rejected, str) and not rejected.strip()):
            print(f'[{i}] 빈 rejected 응답 발견: {worst_model} -> 바로 위 랭킹 응답 사용')
            if len(sorted_models) >= 2:
                second_worst_model, second_worst_score = sorted_models[-2]
                rejected = record[model_response_keys[second_worst_model]]
                worst_model, worst_score = second_worst_model, second_worst_score

        dpo_samples.append({
            "prompt": query,
            "chosen": chosen,
            "rejected": rejected,
            "score_chosen": best_score,
            "score_rejected": worst_score
        })

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_samples, f, ensure_ascii=False, indent=4)
    print(f"✅ 전체 저장 완료: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/ranking_result.json')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/preference_data.json')
    return parser.parse_args()

if __name__ == "__main__":
    main()