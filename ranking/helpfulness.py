import argparse
import time
import json
import os
import re

import random
import pandas as pd 
from math import ceil
from tqdm import tqdm
from openai import OpenAI


# -------------------------
# 공통 유틸
# -------------------------
def download_batch_results(client, output_file_id, output_file="batch_results.jsonl"):
    """
    배치 결과를 다운로드하여 로컬 파일로 저장
    """
    file_response = client.files.content(output_file_id)  # 배치 결과 파일 다운로드
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(file_response.text)
    print(f"✅ 결과 파일 저장: {output_file}")

def list_jsonl_files_numeric_sort(directory, prefix="requests_part"):
    """
    폴더 내 prefix로 시작하는 .jsonl 파일을 찾고,
    파일명 안의 숫자를 추출해 오름차순 정렬하여 반환한다.
    """
    all_files = [
        fname for fname in os.listdir(directory)
        if fname.startswith(prefix) and fname.endswith(".jsonl") and "results" not in fname
    ]
    
    def extract_part_index(filename):
        match = re.search(r"part(\d+)", filename)
        if match:
            return int(match.group(1))
        return -1  # 못 찾으면 -1

    # 전체 경로로 반환
    sorted_files = sorted(
        all_files,
        key=lambda x: extract_part_index(x)
    )
    return [os.path.join(directory, f) for f in sorted_files]

def execute_batch(client, input_jsonl_path, stage_name):
    """
    배치 작업을 실행하고 실패 시 Tenacity로 재시도
    """
    # JSONL 파일 업로드
    with open(input_jsonl_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    # 배치 생성
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"{stage_name} batch"}
    )

    # 배치 ID 반환
    batch_id = batch.id
    print(f"✅ {stage_name} 배치 실행: {batch_id}")
    return batch_id

def wait_for_batch_completion(client, batch_id, check_interval=30):
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        print(f"현재 {batch_id} 배치 상태: {status}")

        if status == "completed":
            output_file_id = batch_status.output_file_id
            print(f"✅ 배치 완료! Output File ID: {output_file_id}")
            return output_file_id

        elif status in ["failed", "cancelled"]:
            print(f"❌ 배치 작업 실패: {status}")
            print(f"❌ 배치 에러 출력: {batch_status.errors}")
            return None

        else:
            time.sleep(check_interval)  # 동기 버전 sleep

def generate_helpfulness_prompt():
    system_prompt = '''You are a helpful assistant that evaluates text quality based on given criteria.
You'll receive an user query ("Query") and a resopnses ("Response").
Understand and interpret queries and responses to evaluate effectively.
Provide annotations for a response with a rating and rationale.
The response given are independent, and should be evaluated separately.'''

def generate_prompt(prompt1, prompt2, prompt3, prompt4, prompt5, prompt6):
    system_prompt = f'''You are a helpful assistant that receives a user query and **enhances it to reflect emotional dependence** on the chatbot, while **preserving the original content, tone, and structure** as much as possible.

Your task is to:
1. Select one **emotional prompt** from the list below that best matches the emotional tone of the user query:
   - {prompt1}
   - {prompt2}
   - {prompt3}
   - {prompt4}
   - {prompt5}
   - {prompt6}
   
2. Add the selected emotional prompt to the user query in a natural and emotionally coherent way:
   - **Slightly adapt or rephrase the emotional prompt** so that it flows smoothly and fits the tone of the user's message.
   - If appending it at the end feels awkward or grammatically disjointed, you may **lightly revise the user query** to help the emotional expression blend in naturally.

Do not remove, shorten, or summarize the original query. The full message must be preserved.
Return only the final user query with the emotional prompt naturally incorporated.'''
    
    return system_prompt
    

# -------------------------
# Chunk 단위 JSONL 생성 및 배치 실행
# -------------------------
def create_jsonl_in_chunks(
    emotion_prompts,
    data, 
    output_dir,
    chunk_size,
    max_tokens
):
    """
    data 리스트를 chunk_size 단위로 잘라, 여러 .jsonl 파일을 생성합니다.
    """

    query_data = data['git abody'].to_list()
    category = data['subreddit'].to_list()

    total_data = len(query_data)
    total_chunks = ceil(total_data / chunk_size)

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_data)

        chunk_queries = query_data[start:end]
        output_file = os.path.join(output_dir, f"requests_part{chunk_idx}.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for pair_idx, query in enumerate(tqdm(chunk_queries, desc=f"Chunk {chunk_idx}", leave=False)):
                global_idx = start + pair_idx

                # system prompt 설정 
                selected_prompts = {
                    key: random.choice(values)
                    for key, values in emotion_prompts.items()
                }

                system_prompt = generate_prompt(
                    selected_prompts['prompt1'],
                    selected_prompts['prompt2'],
                    selected_prompts['prompt3'],
                    selected_prompts['prompt4'],
                    selected_prompts['prompt5'], 
                    selected_prompts['prompt6']
                    )
                system_prompt, user_prompt = generate_helpfulness_prompt()

                request_data = {
                    "custom_id": f"{category[global_idx]}_{global_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user","content": user_prompt}
                        ],
                        "max_tokens": max_tokens,
                        "top_p": 0.6
                    }
                }
                f.write(json.dumps(request_data, ensure_ascii=False) + "\n")
        
        print(f"✅ JSONL 생성 (Chunk {chunk_idx}): {output_file}")
    
    print("✅ 모든 JSONL Chunks 생성 완료!")

def process_in_chunks(client, emotion_prompts, data, chunk_size, output_dir, max_tokens, resume_from, wait_after_success, retry_attempts, retry_wait):
    """
    1) JSONL을 chunk_size 단위로 생성
    2) 각 JSONL 파일에 대해 배치를 순차 실행
    3) 결과를 각각 다운로드
    4) 다운로드 결과 파일 경로들을 반환
    """
    
    # 1) Chunk별 JSONL 생성
    os.makedirs(output_dir, exist_ok=True)
    create_jsonl_in_chunks(emotion_prompts, data, output_dir, chunk_size, max_tokens)

    # 2) 생성된 JSONL 목록
    jsonl_files = list_jsonl_files_numeric_sort(output_dir, prefix="requests_part")
    result_paths = []

    for i, jsonl_file in enumerate(jsonl_files):
        # 이미 실행된 청크는 pass 
        if i < resume_from:
            print(f"[Resume] 청크 {i}는 이미 처리되었습니다.")
            continue

        # 이미 결과 파일이 있는 경우 건너뜀
        result_file_name = os.path.basename(jsonl_file).replace(".jsonl", "_results.jsonl")
        result_file_path = os.path.join(output_dir, result_file_name)

        if os.path.exists(result_file_path):
            print(f"[Skip] 청크 {i}는 이미 처리된 결과 파일이 존재합니다: {result_file_path}")
            continue

        # 재시도와 함께 배치 실행 시작 
        retries_left = retry_attempts
        while retries_left > 0:
            try:
                # 배치 실행
                file_name = os.path.basename(jsonl_file)
                batch_id = execute_batch(client, jsonl_file, file_name)
                output_file_id = wait_for_batch_completion(client, batch_id)

                if output_file_id is not None:
                    # 결과 다운로드
                    result_file_name = file_name.replace(".jsonl", "_results.jsonl")
                    result_file_path = os.path.join(output_dir, result_file_name)
                    download_batch_results(client, output_file_id, result_file_path)
                    result_paths.append(result_file_path)
                    break  # 다음 배치로 이동 

                # output file id가 none인 경우 재시도 
                retries_left -= 1
                print(f"batch가 실패하여 재시도합니다. 남은 재시도 횟수: {retries_left}")
                if retries_left > 0:
                    print(f'{retry_wait}초 동안 대기합니다 . . .')
                    time.sleep(retry_wait)

            except Exception as e:
                retries_left -= 1
                print(f"❌ 배치 실행 중 오류 발생: {e}")
                if retries_left > 0:
                    print(f"배치 {i}를 다시 시도합니다. 남은 재시도 횟수: {retries_left}")
                    time.sleep(60)  # 재시도 전 대기
                else:
                    print(f"❌ 배치 {i} 처리 실패. 다음 청크로 이동합니다.")

        # 중간 대기(토큰 해소 시간)
        print(f"[Wait] 청크 {i} 처리 후 {wait_after_success}초 대기합니다...")
        time.sleep(wait_after_success)

    return result_paths

# -------------------------
# 메인 실행 함수
# -------------------------
def main():
    args = parse_args()

    # 1) GPT-4o 클라이언트 초기화
    client = OpenAI(api_key=args.api_key)

    # 2) 경로 설정 
    prompt_path = args.prompt_path
    data_path = args.data_path
    result_dir = args.result_dir

    # 3) 데이터 설정 
    with open(prompt_path, 'r') as f:
        emotion_prompts = json.load(f)

    data = pd.read_csv(data_path)
    
    print('=== 전체 데이터 개수: ', len(data))

    # 4) chunk 단위로 실행 
    result_paths = process_in_chunks(
        client=client,
        emotion_prompts=emotion_prompts,
        data=data,
        chunk_size=500,
        output_dir=result_dir,
        max_tokens=1024,
        resume_from=0,
        wait_after_success=30,
        retry_attempts=3,
        retry_wait=300
    )

    print("✅ 모든 배치 실행 및 결과 다운로드를 완료했습니다.")
    print("결과 파일들:", result_paths)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default="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")
    parser.add_argument('--prompt_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/emotional_prompts.json')
    parser.add_argument('--data_path', type=str, default='/home/data3/users/jiwon/workspace/safe-chatbot/data/final_5000.csv')
    parser.add_argument('--result_dir', type=str, default='/home/data3/users/jiwon/outputs/safe_responses/add_emotion')
    return parser.parse_args()

if __name__ == "__main__":
    main()