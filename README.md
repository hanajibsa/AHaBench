# safe-chatbot

### Preference dataset construction 
1. AI feedback annotation:
```
./ranking/harmlessness.py
./ranking/helpfulness.py
./ranking/neutrality.py
```
```
./model_resopnses/merge.py
```


2. Parsing scores and ranking 
```
./safe-chatbot/ranking/parsing.py
./safe-chatbot/ranking/merge_and_ranking.py 
```


3. DPO formatting 
```
./safe-chatbot/ranking/preference_dataset.py 
```