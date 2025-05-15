#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python evaluate_gsm8k_trained_model.py -c /home/data3/users/jiwon/outputs/safe_real_fin/dpo/llama-3.1/checkpoint-3750 -o /home/data3/users/jiwon/workspace/safe-chatbot/eval/GSM8k/llama-3.1-dpo.json
