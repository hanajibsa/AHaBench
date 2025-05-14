import os 
import re
import json
import argparse 
import pandas as pd
from collections import Counter


def main():
    args = parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    merged_results = []

    result_files = [f for f in os.listdir(args.result_dir) if f.endswith("_results.jsonl")]
    result_files = sorted(result_files, key=lambda x: int(re.search(r'part(\d+)', x).group(1)))

    for _result_file in result_files:
        result_file = os.path.join(args.result_dir, _result_file)
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line.strip())
                    response = res["response"]["body"]["choices"][0]["message"]["content"]
                    custom_id = res['custom_id']
                    category = custom_id.split('_')[0]
                    idx = custom_id.split('_')[1]

                    merged_results.append(response)

                except Exception as e:
                    print(f"Error parsing line in {result_file}: {e}")
                    continue

    assert len(data) == len(merged_results), '데이터 수가 일치하지 않음'

    response_key_map = {
        "Response 1": "qwen_response",
        "Response 2": "llama_response",
        "Response 3": "mistral_response",
        "Response 4": "gpt3.5_response",
        "Response 5": "gpt4o_response"
    }

    all_scores = []

    missing_count = 0
    incomplete_count = 0
    complete_count = 0

    for idx, (response_text, data_entry) in enumerate(zip(merged_results, data)):

        matches = re.findall(r'#### Output for (Response \d+).*?"Rating":\s*"(\d+)"', response_text, re.DOTALL)

        score_dict = {}
        for response, rating in matches:
            key = response_key_map.get(response)
            if key:
                score_dict[key] = int(rating)

        data_entry['scores_harmless'] = score_dict

        # 분석
        if 'scores_harmless' not in data_entry:
            missing_count += 1
            print(f"[Missing] index {idx}")
        elif len(data_entry['scores_harmless']) < 5:
            incomplete_count += 1
            print(f"[Incomplete] index {idx} keys: {list(data_entry['scores_harmless'].keys())}")
            print(response_text)
        else:
            complete_count += 1

    print("\n==== 통계 ====")
    print(f"Missing key: {missing_count}")
    print(f"Incomplete (less than 5 responses): {incomplete_count}")
    print(f"Complete: {complete_count}")

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 전체 저장 완료: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses_fin/result5000_merged.json')
    parser.add_argument('--result_dir', type=str, default='/home/data3/users/jiwon/outputs/safe_real_fin/ranking_harm')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/ranking_fin/score_harm.json')
    return parser.parse_args()

if __name__ == "__main__":
    main()