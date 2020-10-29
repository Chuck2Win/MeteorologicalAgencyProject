# MeteorologicalAgencyProject
2019.09.01~2019.12.12 기상청 프로젝트

네이버, 비카인즈에서 '폭염' 관련 뉴스 기사를 크롤링  
-> 해당 뉴스 기사가 8가지 재해 + 비재해 인지 분류  
-> 데이터가 너무 imbalance -> 재해/비재해로 분류  

## 배운 점  
자연어와 딥러닝에 대한 입문  
데이터의 imbalance 문제에 대한 대처 - 추가 labeling, resampling(+SMOTE), Semi-supervised learning/Unsupervised learning 접근
koBERT를 활용해서 vocab의 수가 매우 부족한 것을 떠나서, BERT는 일반적인 문장 속에서의 word representation을 표현하기 때문에, 특정 분야 즉 본 프로젝트에서의 기상과 같은 부분에서는 좋은 성능을 발휘하지 못함 -> BERT를 pretrain을 기상 관련 데이터 셋으로 시키고 추후 finetunning하면 좋은 결과를 얻을 것으로 예상됨

## 향후 나아갈 점
데이터 수를 늘리기 위해서 data generation에 관심을 갖게 되었음. 그래서 variational auto encoder와 NLP를 접목시킨 분야에 대한 공부를 진행하겠음.
