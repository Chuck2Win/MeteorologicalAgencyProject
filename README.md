# 빅데이터를 이용한 재해기상정보 활용 연구
기간 : 2019.09.01~2019.12.12 
주체 : 국립기상과학원 프로젝트  
본인은 이 프로젝트에서 **중복기사** 제거하기 및 **BERT를 활용한 Classifier 만들기**를 수행하였음.  

# Data Description(세부 EDA,Preprocessing 참조)    
네이버, 비카인즈에서 '폭염' 관련 뉴스 기사를 크롤링(14만건) -> 중복기사제거(11만건)      
이 중 8,048건을 라벨링 -> 전처리 후 피해 관련 1,297건, 피해 비관련 6,701건    
초기 분류 - 8가지 재해와 비재해  
Imbalance Issue로 재해/비재해 binary classification 문제로 환원  

# Model  
Kobert+classifier  
![model](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/model.png)  
Token화 된 제목+본문과 길이를 넣어줌  

# 학습 방식  
Early Stopping 방식 적용   
model 3는 2021.2.19 에 추가됨.(후속연구)      

## Augmented dataset 형성(후속연구)    
본인의 논문인 "Soley Transformer based Variational Auto Encoder For Sentence Generation"의 idea 활용  
데이터 생성하는 코드는 본인 논문의 github에 있음(학위 논문 제출 후 공개 예정)  
생성된 재해 관련 기사는 Greedy Decoding으로 생성함(Coverage mechanism 적용 X)  
![augmented data](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/table6.jpg)  

# 결과  
| classifier  | Data set   | Sampling           | Train data set | Val data set | Test data set |
| ----------- | ---------- | ------------------ | -------------- | ------------ | ------------- |
| classifier1 | Imbalanced | Random  Sampling   | Imbalanced     | Imbalanced   | Imbalanced    |
| classifier2  | Imbalanced | Weighted  Sampling | balanced       | balanced     | Imbalanced    |
| classifier3 | Augmented  | Random  Sampling   | balanced       | balanced     | Imbalanced    |  

## AUC, ROC
![model](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/fig10.png)  

## Acc,Precision,Recall,F1
|              |      | precision | recall | f1     | support |
| ------------ | ---- | --------- | ------ | ------ | ------- |
| classifier 1 | 0    | 0.9185    | 0.9501 | 0.9340 | 842     |
|              | 1    | 0.6744    | 0.5506 | 0.6063 | 158     |
| classifier 2 | 0    | 0.9766    | 0.6948 | 0.8119 | 842     |
|              | 1    | 0.3591    | 0.9114 | 0.5152 | 158     |
| classifier 3 | 0    | 0.9309    | 0.9442 | 0.9375 | 842     |
|              | 1    | 0.6781    | 0.6266 | 0.6513 | 158     |

|              | accuracy | cross entropy | AUC |
| ------------ | -------- | ------------- | ------------- |
| classifier 1 | 0.8870   | 0.03183       |0.863|
| classifier 2 | 0.7290   | 0.07741       |0.882|
| classifier 3 | 0.8940   | 0.03549       |0.888|
  
데이터를 생성해서 추가해서 학습한 모델이 가장 좋은 결과를 낳게됨  
minority class에 대해서 precision은 f1 score은 0.045(7.4%) 상승, accuracy는 0.007(0.78%)상승, cross entropy는 증가함.
AUC 역시 0.888로 제일 높음  
# 환경
google colab에서 진행함.   
추가적으로 필요한 library  
1. transformers
2. kobert-transformers
3. kss
4. sentencepiece

# 사용방법
## 전처리
! python3 preprocessing.py --data_file (위치) --return_file (저장할 위치)  
(train test validation split은 직접 진행하면 됨, 내가 임의로 분류한 train val test set는 data 폴더에 있음.)    
## train
! python3 train.py --model (모델 : Augmentation, None--min_model, WeightedRandomSample) --min_model (모델 저장명) --train_file (train file) --val_file (val file) --generated_sentence_file (생성시킨 파일의 위치)

## inference
! python3 inference.py --data_file (inference하고 싶은 전처리된 파일의 위치) --min_model (사용하고자 하는 모델의 위치) --predict False predict를 True로 하면, 해당 기사의 예측값을 원하는 저장 위치로 보냄. predict가 False라면 예측하고 loss와 acc 등을 계산(label이 있는 데이터에 활용하면 됨) 

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
