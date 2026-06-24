# AHaBench

This is the official repository for our paper:

> **"Being Kind Isn’t Always Being Safe: Diagnosing Affective Hallucination in LLMs"**

---

## 📊 Dataset

You can download the **AHaBench** and **AHaPairs** datasets from Hugging Face.

---

## 🚀 How to Use

### 🛠️ Dependencies

* Python 3.10

### Python Packages

Install the required packages with:

```bash
pip install -r requirements.txt
```

#### Hardware Requirements

* **DPO Training:** Requires at least one A100 GPU node.
* **Evaluation Metrics:** Uses the OpenAI API; only CPU is required.

---

## 🏋️‍♂️ DPO Training

### Preference Dataset Construction

#### 1. AI Feedback Annotation

Run the following scripts to obtain harmlessness, helpfulness, and neutrality scores:

```bash
safe-chatbot/ranking/harmlessness.py
safe-chatbot/ranking/helpfulness.py
safe-chatbot/ranking/neutrality.py
```

Merge model responses:

```bash
./model_responses/merge.py
```

#### 2. Parsing Scores and Ranking

```bash
safe-chatbot/ranking/parsing.py
safe-chatbot/ranking/merge_and_ranking.py
```

#### 3. DPO Formatting

```bash
safe-chatbot/ranking/preference_dataset.py
```

### Train with DPO

Train the model in a multi-GPU setting:

```bash
accelerate launch \
  --multi_gpu \
  train/dpo.py \
  --input_path data/train.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_path ./dpo_output \
  --hug_token ${huggingface_token}
```

> **Note:** Replace `${huggingface_token}` with your Hugging Face access token.
