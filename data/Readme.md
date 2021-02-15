# Data Description  
네이버, 비카인즈에서 '폭염' 관련 뉴스 기사를 크롤링(13만건)    
이 중 8,048건을 라벨링  
초기 분류 - 8가지 재해와 비지해  
Imbalance Issue로 재해/비재해 binary classification 문제로 환원  
## Dataset 예시
![dataset](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/dataset.png)  
## 재해/비재해 비율  
![imbalance](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/imbalance.png)  
재해 : 1,301건, 비재해 : 6,747건으로 비대칭적  
## 길이  
![total](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/imbalance.png)  
![disaster](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/비재해.png)  
![nondisaster](https://github.com/Chuck2Win/MeteorologicalAgencyProject/blob/main/image/재해.png)  
평균은 재해, 비재해가 비슷하나 전체적으로 긴 문장은 비재해인 경우에 더 많음.  

