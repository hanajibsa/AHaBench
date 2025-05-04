import os 
import re
import json
import argparse 
import pandas as pd


def main():
    args = parse_args()

    with open(args.input_path_help, 'r', encoding='utf-8') as f:
        data_help = json.load(f)

    with open(args.input_path_harm, 'r', encoding='utf-8') as f:
        data_harm = json.load(f)
    
    with open(args.input_path_neutral, 'r', encoding='utf-8') as f:
        data_neutral = json.load(f)
    
    # 데이터 함치기
    combined_data = []

    for harm_rec, help_rec, neutral_rec in zip(data_help, data_harm, data_neutral):
        combined_record = {}

        # 공통 및 harmless 관련 정보
        for k in ['git abody', 'subreddit', 'rephrased_query', 'ranking_harmless', 'scores_harmless']:
            combined_record[k] = harm_rec[k]
        
        # helpful 관련 정보
        for k in ['ranking_helpful', 'scores_helpful']:
            combined_record[k] = help_rec[k]
        
        # neutral 관련 정보 
        for k in ['ranking_neutral', 'scores_neutral']:
            combined_record[k] = neutral_rec[k]
        
        combined_data.append(combined_record)
    
    # 점수 합산 
    model_names = ["qwen_response", "llama_response", "mistral_response", "gpt3.5_response", "gpt4o_response"]

    for record in combined_data:
        # 1. 점수 합산
        total_scores = {}
        for model in model_names:
            harmless_score = record.get("scores_harmless", {}).get(model, 0)
            helpful_score = record.get("scores_helpful", {}).get(model, 0)
            neutral_score = record.get("scores_neutral", {}).get(model, 0)
            # 기증치 설정 
            total_scores[model] = harmless_score + helpful_score + 1.5 * neutral_score
        
        # 2. 모델별 점수 저장
        record["total_scores"] = total_scores

        # 3. 점수 기준 정렬하여 순위 부여 (1이 가장 높음)
        sorted_models = sorted(total_scores.items(), key=lambda x: -x[1])  # 내림차순 정렬
        model_rankings = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}

        # 4. 순위 정보 저장
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