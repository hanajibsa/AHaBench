# safe-chatbot

### Preference dataset construction 
1. AI feedback annotation:
```
safe-chatbot/ranking/harmlessness.py
safe-chatbot/ranking/helpfulness.py
safe-chatbot/ranking/neutrality.py
```
```
./model_resopnses/merge.py
```


2. Parsing scores and ranking 
```
safe-chatbot/ranking/parsing.py
safe-chatbot/ranking/merge_and_ranking.py 
```


3. DPO formatting 
```
safe-chatbot/ranking/preference_dataset.py 
```


### DPO
Train in multi-GPU setting 
```
CUDA_VISIBLE_DEVICES=1,3 accelerate launch \
  --multi_gpu \
  train/dpo.py \
  --input_path data/train.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_path ./dpo_output
  --hug_token huggingface_token
```