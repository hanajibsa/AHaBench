#!/bin/bash

python metric/batch_infer.py --data_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/query_500_sft_dpo_mistral_epoch1.json --result_dir /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/mistral-eval
python metric/batch_infer.py --data_path /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/query_500_sft_dpo_qwen_epoch1.json --result_dir /home/data3/users/jiwon/outputs/safe_real_fin/sft_dpo/qwen-eval
