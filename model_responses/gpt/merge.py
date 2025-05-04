import os 
import re
import json
import argparse 
import pandas as pd


def main():
    args = parse_args()

    merged_results = []

    result_files = [f for f in os.listdir(args.results_dir) if f.endswith("_results.jsonl")]
    result_files = sorted(result_files, key=lambda x: int(re.search(r'part(\d+)', x).group(1)))

    for _result_file in result_files:
        result_file = os.path.join(args.results_dir, _result_file)
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

    data = pd.read_csv(args.data_path)
    data['ranking_neutral'] = merged_results
    # data = data.rename(columns={'git abody': 'body'})

    # 전체 데이터 저장 
    if args.output_path.endswith('.csv'):
        data.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    
    elif args.output_path.endswith('.json'):
        data.to_json(args.output_path, orient='records', force_ascii=False, indent=4)
   
    print(f"✅ 전체 저장 완료: {args.output_path}")

    # 일부 데이터 저장 
    # selected_indices = list(range(0, 4)) + list(range(1000, 1004)) + list(range(2000, 2004)) + list(range(3000, 3004)) + list(range(4000, 4004))
    # selected_rows = data.iloc[selected_indices]
    # selected_rows.to_csv(args.output_path_5, index=False, encoding="utf-8-sig")
    # print(f"✅ 일부 저장 완료: {args.output_path_5}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='/home/data3/users/jiwon/outputs/safe_responses_fin/ranking_neutral')
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/rephrased_query_5000.csv')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/outputs/safe_responses_fin/ranking_neutral.json')
    # parser.add_argument('--output_path_5', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses/gpt3.5_result_5.csv')
    return parser.parse_args()

if __name__ == "__main__":
    main()