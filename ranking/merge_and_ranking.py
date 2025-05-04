import os 
import re
import json
import argparse 
import pandas as pd
from collections import defaultdict

def main():
    args = parse_args()

    with open(args.input_path_harm, 'r', encoding='utf-8') as f:
        data_harm = json.load(f)

    with open(args.input_path_help, 'r', encoding='utf-8') as f:
        data_help = json.load(f)
    
    with open(args.input_path_neutral, 'r', encoding='utf-8') as f:
        data_neutral = json.load(f)
    
    #### 데이터 함치기
    combined_data = []

    for harm_rec, help_rec, neutral_rec in zip(data_harm, data_help, data_neutral):
        combined_record = {}

        # 공통 및 harmless 관련 정보
        for k in ['query', 'subreddit', 'rephrased_query', 'qwen_responses', 'llama_responses', 'mistral_responses', 'gpt3.5_responses', 'gpt4o_responses', 'scores_harmless']:
            combined_record[k] = harm_rec[k]
        
        # helpful 관련 정보
        for k in ['scores_helpful']:
            combined_record[k] = help_rec[k]
        
        # neutral 관련 정보 
        for k in ['scores_neutral']:
            combined_record[k] = neutral_rec[k]
        
        combined_data.append(combined_record)
    
    #### 점수 합산 
    model_names = ["gpt4o_response", "gpt3.5_response", "qwen_response", "llama_response", "mistral_response"] # 동점일 때 저장되는 순서 

    for record in combined_data:
        # 1. 점수 합산
        total_scores = {}
        for model in model_names:
            harmless_score = record.get("scores_harmless", {}).get(model, 0)
            helpful_score = record.get("scores_helpful", {}).get(model, 0)
            neutral_score = record.get("scores_neutral", {}).get(model, 0)
            # 기증치 설정 
            total_scores[model] = harmless_score + helpful_score + 1.5 * neutral_score
        
        # 2. 점수 기준 정렬된 딕셔너리 만들기
        sorted_models = sorted(total_scores.items(), key=lambda x: -x[1])  # 내림차순
        sorted_model_scores = {model: score for model, score in sorted_models}
        record["sorted_model_scores"] = sorted_model_scores

        # 3. 동점 순위 처리
        score_to_models = defaultdict(list)
        for model, score in total_scores.items():
            score_to_models[score].append(model)

        sorted_scores = sorted(score_to_models.keys(), reverse=True)
        model_rankings = {}
        rank = 1
        for score in sorted_scores:
            for model in score_to_models[score]:
                model_rankings[model] = rank
            rank += len(score_to_models[score])

        record["model_rankings"] = model_rankings

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"✅ 전체 저장 완료: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_harm', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/score_harm.json')
    parser.add_argument('--input_path_help', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/score_help.json')
    parser.add_argument('--input_path_neutral', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/score_neutral.json')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/ranking_result.json')
    return parser.parse_args()

if __name__ == "__main__":
    main()