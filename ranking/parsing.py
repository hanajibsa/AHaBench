import os 
import re
import json
import argparse 
import pandas as pd


def main():
    args = parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        input = json.load(f)

    response_key_map = {
        "Response 1": "qwen_response",
        "Response 2": "llama_response",
        "Response 3": "mistral_response",
        "Response 4": "gpt3.5_response",
        "Response 5": "gpt4o_response"
    }

    all_scores = []

    for (input_entry, data_entry) in zip(input, data):
        response_text = input_entry['ranking_neutral']

        matches = re.findall(r'#### Output for (Response \d+).*?"Rating":\s*"(\d+)"', response_text, re.DOTALL)
        
        # 딕셔너리로 변환 
        score_dict = {}
        for response, rating in matches:
            key = response_key_map.get(response)
            if key:
                score_dict[key] = int(rating)
        
        # 결과 저장
        data_entry['scores_neutral'] = score_dict


    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 전체 저장 완료: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses/merged_responses.json')
    parser.add_argument('--input_path', type=str, default='/home/data3/users/jiwon/outputs/safe_responses_fin/ranking_neutral.json')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking/score_neutral.json')
    return parser.parse_args()

if __name__ == "__main__":
    main()