# 기상청 재해 뉴스 기사 분류

## 1. Model  
BERT+classifier  

## Imbalance 에 대처하기 위한 방안  
### 1) Data Augmentation  
내 논문에 의거해서 Data Augmentation 진행  
### 2) Random Sampling  
비재해의 경우가 많아서 Batch 생성시 재해와 비재해 비율을 비슷하게 Sampling해서 학습 진행  

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
Accuracy : 0.80
Precision : 0.43
Recall : 0.74

추가적으로 TF-IDF를 통해서 lstm 혹은 Random Forest 진행시 성능이 너무 안좋아서 기술 안함.

