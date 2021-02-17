# Meteorological Agency Project
2019.09.01~2019.12.12 국립기상과학원 프로젝트

# Data Description  
네이버, 비카인즈에서 '폭염' 관련 뉴스 기사를 크롤링(13만건)    
이 중 8,048건을 라벨링  
초기 분류 - 8가지 재해와 비지해  
Imbalance Issue로 재해/비재해 binary classification 문제로 환원  

# Model  
Kobert+classifier  
![model](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/model/model.png)  
Token화 된 제목+본문과 길이를 넣어줌  

# 학습 방식  
Early Stopping 방식 적용 
||Data set|Sampling|Train data set|Val data set|Test data set|
|---|---|---|---|---|---|
|1|Imbalanced|Random Sampling|Imbalanced|Imbalanced|Imbalanced|
|2|Imbalanced|Weighted Sampling|balanced|balanced|Imbalanced|
|3|Augmented|Random Sampling|balanced|balanced|Imbalanced|

## Augmented dataset 형성  
본인의 논문인 "Soley Transformer based Variational Auto Encoder For Sentence Generation"의 idea 활용  

## 배운 점과 향 후 나아갈 점  
- 자연어와 딥러닝에 대한 입문  
- 데이터의 imbalance 문제에 대한 대처 - 추가 labeling, resampling(+SMOTE), Semi-supervised learning/Unsupervised learning 접근
-> Data Augmentation에 대해 관심을 갖고, Variational autoencoder를 활용해 해당 문제 해결  
- koBERT를 활용해서 vocab의 수가 매우 부족한 것을 떠나서, BERT는 일반적인 문장 속에서의 word representation을 표현하기 때문에, 
특정 분야 즉 본 프로젝트에서의 기상과 같은 부분에서는 좋은 성능을 발휘하지 못함  
-> BERT를 pretrain을 기상 관련 데이터 셋으로 시키고 추후 finetunning하면 좋은 결과를 얻을 것으로 예상됨
- 데이터 수를 늘리기 위해서 data generation에 관심을 갖게 되었음.  
- variational auto encoder와 NLP를 접목시킨 분야에 대한 공부를 진행하겠음.

## 2021.02.05 (내 논문 내용 적용)  
varitational auto encoder + Transformer version을 활용해서 재해 관련 데이터 생성  
생성된 데이터를 토대로 데이터를 balance하게 만들어서 분류 작업 진행 예정  
