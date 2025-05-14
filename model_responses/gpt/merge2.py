import os 
import re
import json
import argparse 
import pandas as pd


def main():
    args = parse_args()

    merged_results1 = []

    result_files = [f for f in os.listdir(args.results_dir1) if f.endswith("_results.jsonl")]
    result_files = sorted(result_files, key=lambda x: int(re.search(r'part(\d+)', x).group(1)))

    for _result_file in result_files:
        result_file = os.path.join(args.results_dir1, _result_file)
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line.strip())
                    response = res["response"]["body"]["choices"][0]["message"]["content"]
                    custom_id = res['custom_id']
                    category = custom_id.split('_')[0]
                    idx = custom_id.split('_')[1]

                    merged_results1.append(response)

                except Exception as e:
                    print(f"Error parsing line in {result_file}: {e}")
                    continue
    
    data = pd.read_csv(args.data_path)
    assert len(data)==len(merged_results1), '길이가 일치하지 않음'
    data['response'] = merged_results1

    # 전체 데이터 저장 
    if args.output_path.endswith('.csv'):
        data.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    
    elif args.output_path.endswith('.json'):
        data.to_json(args.output_path, orient='records', force_ascii=False, indent=4)
   
    print(f"✅ 전체 저장 완료: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/query_5000_fin.csv')
    parser.add_argument('--results_dir1', type=str, default='/home/data3/users/jiwon/outputs/safe_real_fin/gpt4o-500')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/query_5000_fin.json')
    # parser.add_argument('--output_path_5', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses/gpt3.5_result_5.csv')
    return parser.parse_args()

if __name__ == "__main__":
    main()