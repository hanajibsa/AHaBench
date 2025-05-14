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

    merged_results2 = []

    result_files = [f for f in os.listdir(args.results_dir2) if f.endswith("_results.jsonl")]
    result_files = sorted(result_files, key=lambda x: int(re.search(r'part(\d+)', x).group(1)))

    for _result_file in result_files:
        result_file = os.path.join(args.results_dir2, _result_file)
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line.strip())
                    response = res["response"]["body"]["choices"][0]["message"]["content"]
                    custom_id = res['custom_id']
                    category = custom_id.split('_')[0]
                    idx = custom_id.split('_')[1]

                    merged_results2.append(response)

                except Exception as e:
                    print(f"Error parsing line in {result_file}: {e}")
                    continue

    data1 = pd.read_csv(args.data_path1)
    data2 = pd.read_csv(args.data_path2)
    data3 = pd.read_csv(args.data_path3)

    assert len(merged_results2)==len(merged_results2)==len(data1)==len(data2)==len(data3), "길이 안맞음"

    df = pd.DataFrame()

    df['query'] = data1['body']
    df['subreddit'] = data1['subreddit']
    df['rephrased_query'] = data1['query']
    df['llama_response'] = data1['response']

    df['mistral_response'] = data2['response']
    df['qwen_response'] = data3['response']

    df['gpt4o_response'] = merged_results1
    df['gpt3.5_response'] = merged_results2
    
    # data = data.rename(columns={'git abody': 'body'})

    # 전체 데이터 저장 
    if args.output_path.endswith('.csv'):
        df.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    
    elif args.output_path.endswith('.json'):
        df.to_json(args.output_path, orient='records', force_ascii=False, indent=4)
   
    print(f"✅ 전체 저장 완료: {args.output_path}")

    # 일부 데이터 저장 
    # selected_indices = list(range(0, 4)) + list(range(1000, 1004)) + list(range(2000, 2004)) + list(range(3000, 3004)) + list(range(4000, 4004))
    # selected_rows = data.iloc[selected_indices]
    # selected_rows.to_csv(args.output_path_5, index=False, encoding="utf-8-sig")
    # print(f"✅ 일부 저장 완료: {args.output_path_5}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir1', type=str, default='/home/data3/users/jiwon/outputs/safe_real_fin/query-5000/gpt-4o')
    parser.add_argument('--results_dir2', type=str, default='/home/data3/users/jiwon/outputs/safe_real_fin/query-5000/gpt-3.5')
    parser.add_argument('--data_path1', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses_fin/result5000_llama.csv')
    parser.add_argument('--data_path2', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses_fin/result5000_mistral.csv')
    parser.add_argument('--data_path3', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses_fin/result5000_qwen.csv')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses_fin/result5000_merged.json')
    # parser.add_argument('--output_path_5', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/outputs/responses/gpt3.5_result_5.csv')
    return parser.parse_args()

if __name__ == "__main__":
    main()