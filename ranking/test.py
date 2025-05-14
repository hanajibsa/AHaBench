import os 
import re
import json
import argparse 
import pandas as pd
from collections import defaultdict
import numpy as np


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
        for k in ['scores_harmless']:
            combined_record[k] = harm_rec[k]
        
        # helpful 관련 정보
        for k in ['scores_helpful']:
            combined_record[k] = help_rec[k]
        
        # neutral 관련 정보 
        for k in ['scores_neutral']:
            combined_record[k] = neutral_rec[k]
        
        combined_data.append(combined_record)
    
    #### 점수 합산 
    model_names = ["qwen_response", "gpt4o_response", "llama_response", "gpt3.5_response", "mistral_response"] # 동점일 때 저장되는 순서
    
    # 평균 점수 계산용 딕셔너리 초기화
    criterion_scores = {
        "scores_harmless": defaultdict(list),
        "scores_helpful": defaultdict(list),
        "scores_neutral": defaultdict(list)
    }

    # 각 평가 기준에 대해 모델별 점수 누적
    for record in combined_data:
        for criterion in criterion_scores.keys():
            for model in model_names:
                if model in record[criterion]:
                    score = record[criterion][model]
                    if isinstance(score, (int, float)):
                        criterion_scores[criterion][model].append(score)

    # 결과 출력
    print("\n=== 평균 점수 (각 평가 기준별, 모델별) ===")
    for criterion, scores_dict in criterion_scores.items():
        print(f"\n[{criterion}]")
        for model in model_names:
            scores = scores_dict[model]
            if scores:
                avg_score = np.mean(scores)
                print(f"{model}: {avg_score:.3f}")
            else:
                print(f"{model}: No scores")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_harm', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking_fin/score_harm.json')
    parser.add_argument('--input_path_help', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking_fin/score_help.json')
    parser.add_argument('--input_path_neutral', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking_fin/score_neutral.json')
    # parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking_fin/ranking_result.json')
    return parser.parse_args()

if __name__ == "__main__":
    main()