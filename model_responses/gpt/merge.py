import os 
import re
import json
import argparse 


def main():
    args = parse_args()

    merged_results = []

    result_files = [f for f in os.listdir(args.results_path) if f.endswith("_results.jsonl")]
    result_files = sorted(result_files, key=lambda x: int(re.search(r'part(\d+)', x).group(1)))

    for _result_file in result_files:
        result_file = os.path.join(args.results_path, _result_file)
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line.strip())
                    response = res["response"]["body"]["choices"][0]["message"]["content"]
                    custom_id = res['custom_id']
                    category = custom_id.split('_')[0]
                    idx = custom_id.split('_')[1]

                    merged_results.append({
                        "idx": idx,
                        "image_id": custom_id,
                        "response": response,
                        "category": category
                    })

                except Exception as e:
                    print(f"Error parsing line in {result_file}: {e}")
                    continue
    
    with open(args.output_path, "w", encoding="utf-8") as out_file:
        json.dump(merged_results, out_file, ensure_ascii=False, indent=2)

    print(f"✅ JSON 저장 완료: {args.output_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='/home/data3/users/jiwon/outputs/safe_responses/gpt-4o')
    parser.add_argument('--output_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/system_prompts_5000.json')
    return parser.parse_args()

if __name__ == "__main__":
    main()