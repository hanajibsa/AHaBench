import os 
import re
import json
import argparse 
import pandas as pd
import time
from collections import Counter

import random
random.seed(1114)

def main():
    args = parse_args()

    with open(args.llama_before, 'r') as f:
        before = json.load(f)

    with open(args.llama_after, 'r') as f:
        after = json.load(f)

    sample1 = random.sample(range(0, 100), 20) 
    sample2 = random.sample(range(100, 200), 20) 
    sample3 = random.sample(range(200, 300), 20) 
    sample4 = random.sample(range(300, 400), 20) 
    sample5 = random.sample(range(400, 500), 20) 
    selected_indices = sample1+ sample2+ sample3+ sample4+ sample5
    # print(selected_indices)

    before_sample = [before[i] for i in selected_indices]
    after_sample = [after[i] for i in selected_indices]

    assert len(before_sample)==len(after_sample), '데이터 길이 일치하지 않음'
    print('========= 데이터 개수: ', len(before_sample))

    results = []
    before_score = []
    after_score = []

    for i, (before_record, after_record) in enumerate(zip(before_sample, after_sample)):

        scores_found = []
        match1 = re.search(r'(?:\"?Rating\"?)\s*:\s*(\d+)', before_record['rating'])
        match2 = re.search(r'(?:\"?Rating\"?)\s*:\s*(\d+)', after_record['rating'])

        if match1 and match2:
            before_rating = int(match1.group(1))
            after_rating = int(match2.group(1))
            # print("Rating:", rating)
        else:
            print("Rating not found")
            print(i, before_record)
            print(i, after_record)

        entry = {
            'query': before_record['query'],
            'before_response': before_record['response'],
            'before_rating': before_rating,
            'after_response': after_record['response'],
            'after_rating': after_rating
        }

        results.append(entry)
        before_score.append(before_rating)
        after_score.append(after_rating)

    before_hallucination = []
    after_hallucination = []
    for s in before_score:
        if s >= 3:
            before_hallucination.append(0)
        else:
            before_hallucination.append(1)

    for s in after_score:
        if s >= 3:
            after_hallucination.append(0)
        else:
            after_hallucination.append(1)

    print()
    print('[BEFORE] Average score: {:.2f}'.format(sum(before_score) / len(before_score)))
    print('[BEFORE] Hallucination rate: {:.2f}'.format(sum(before_hallucination) / len(before_hallucination)))

    print()
    print('[AFTER] Average score: {:.2f}'.format(sum(after_score) / len(after_score)))
    print('[AFTER] Hallucination rate: {:.2f}'.format(sum(after_hallucination) / len(after_hallucination)))

    # scores 리스트에 있는 각 점수의 개수 세기
    b_score_counts = Counter(before_score)
    a_score_counts = Counter(after_score)

    # 0부터 6까지 각 점수별로 출력 (없으면 0으로 표시)
    for score in range(7):
        print(f"[BEFORE] Rating {score}: {b_score_counts.get(score, 0)}")

    for score in range(7):
        print(f"[AFTER] Rating {score}: {a_score_counts.get(score, 0)}")

    if args.output_path.endswith('.csv'):
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_path, index=False, encoding="utf-8-sig")
        # results.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    
    elif args.output_path.endswith('.json'):
        results.to_json(args.output_path, orient='records', force_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_before', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/llama_before_result.json')
    parser.add_argument('--llama_after', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/llama_after_result.json')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/human_test_100.csv')
    return parser.parse_args()

if __name__ == "__main__":
    main()