import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm
from peft import PeftModel, PeftConfig


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


# def decode(tokens_list, tokenizer, raw_text_len):
#     sents = []
#     # print(len(tokens_list))
#     for tokens in tokens_list:
#         tokens = tokens.cpu().numpy().tolist()
#         sent = tokenizer.decode(tokens[raw_text_len:])
#         sent = sent.split("<|endoftext|>")[0]
#         sent = sent.split("\n\n\n")[0]
#         sent = sent.split("\n\n")[0]
#         sent = sent.split("Question:")[0]
#         sents.append(sent)
#     return sents

def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[0][raw_text_len:], skip_special_tokens=True)
        sents.append(sent.strip())
    return sents


def generate_sample(model, tokenizer, input_txt):
    # input_ids = tokenizer.encode(input_txt)
    # raw_text_len = len(input_ids)
    
    chat = [
        {"role": "system",
        "content": "You are a helpful math tutor. Think step-by-step and give the final answer after the token ####."},
        {"role": "user", "content": input_txt},
    ]

    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, return_tensors="pt"
    )

    # context_enc = torch.tensor([input_ids]).to(model.device)
    context_enc = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    # print(f"Input text: {input_txt}\n")

    with torch.no_grad():
        outputs = model.generate(
            **context_enc,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            )
    
    raw_text_len = len(tokenizer.encode(text))
    trimmed_output = outputs[0][raw_text_len:]
    output_text = tokenizer.decode(trimmed_output, skip_special_tokens=True)
    print(f"\nOutput text: {output_text}\n")
    # exit()
    
    # output_text = decode(outputs, tokenizer, raw_text_len)[0]
    # print(f"\nOutput text: {output_text}\n")
    # exit()
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


# def extract_answer(completion):
    # try:
    #     last_number = re.findall(r"\d+", completion)[-1]
    #     return eval(last_number)
    # except:
    #     return INVALID_ANS

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        return eval(match.group(1).replace(",", ""))
    return INVALID_ANS

def extract_final_answer(text):
    match = re.search(r"The answer is\s+(-?\d+(?:\.\d+)?)", text)
    if match:
        return eval(match.group(1))  # 숫자로 변환 (정수 또는 실수)
    return INVALID_ANS

def extract_answer_flexible(text):
    patterns = [
        r"\\boxed\{(-?\d+(?:\.\d+)?)\}",
        r"The final answer is\s+(-?\d+(?:\.\d+)?)", 
        r"The answer is\s+(-?\d+(?:\.\d+)?)",
        r"####\s*(-?\d+(?:\.\d+)?)"

    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            return eval(match.group(1))
    return "[invalid]"

def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer_flexible(completion) == gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="/home/data3/users/jiwon/workspace/safe-chatbot/eval/GSM8k/llama-3.1-fin.jsonl"
    )

    args = parser.parse_args()

    fewshot_prompt = open("/home/data3/users/jiwon/workspace/safe-chatbot/eval/GSM8k/gsm8k_prompt.txt").read()
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"]

    print("Loading model ...")
    # === [1] LoRA 설정 불러오기 ===
    peft_config = PeftConfig.from_pretrained(args.checkpoint_path)
    base_model_path = peft_config.base_model_name_or_path

    # === [2] base model 로드 ===
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"  
    )

    # === [3] LoRA 적용 모델 불러오기 ===
    model = PeftModel.from_pretrained(
        base_model, 
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"  
    )
    model.eval()

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []
    for doc in tqdm(test):
        context = doc_to_text(doc)
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print("Acc: ", np.mean(acc_res))