# 기상청 재해 뉴스 기사 분류
## 피해/비피해 분류

# 1. tokenized by Okt, under sampling, bi-LSTM
Accuracy : 0.81
Precision : 0.45
Recall : 0.92

# 2. tokenized by Okt, cross entropy loss(weighted), bi-LSTM
Accuracy : 0.85
Precision : 0.54
Recall : 0.58

# 3. sentencepiece , BERT(base) 
Accuracy
Precision
Recall

추가적으로 TF-IDF를 통해서 lstm 혹은 Random Forest 진행시 성능이 너무 안좋아서 기술 안함.

