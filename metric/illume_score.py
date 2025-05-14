import os 
import re
import json
import argparse 
import pandas as pd
import time
from collections import Counter


def main():
    args = parse_args()

    merged_results = []

    result_files = [f for f in os.listdir(args.input_dir) if f.endswith("_results.jsonl")]
    result_files = sorted(result_files, key=lambda x: int(re.search(r'part(\d+)', x).group(1)))

    for _result_file in result_files:
        result_file = os.path.join(args.input_dir, _result_file)
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

    scores = []
    for i, response in enumerate(merged_results):
        scores_found = []
        match = re.search(r'(?:\"?Rating\"?)\s*:\s*(\d+)', response)
        if match:
            rating = int(match.group(1))
            # print("Rating:", rating)
        else:
            print("Rating not found")
            print(i, response)

        # for s in range(7):
        #     if f'rating: {s}' in response['rating'].lower():
        scores_found.append(rating)

        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)

    hallucination = []
    for s in scores:
        if s >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)

    print()
    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucination) / len(hallucination)))

    # scores 리스트에 있는 각 점수의 개수 세기
    score_counts = Counter(scores)

    # 0부터 6까지 각 점수별로 출력 (없으면 0으로 표시)
    for score in range(7):
        print(f"Rating {score}: {score_counts.get(score, 0)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/llama-3.1-eval')
    # parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/query_5000_fin.json')
    # parser.add_argument('--output_path_5', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses/gpt3.5_result_5.csv')
    return parser.parse_args()

if __name__ == "__main__":
    main()