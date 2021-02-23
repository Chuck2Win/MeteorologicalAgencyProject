# 결과  
| classifier  | Data set   | Sampling           | Train data set | Val data set | Test data set |
| ----------- | ---------- | ------------------ | -------------- | ------------ | ------------- |
| classifier1 | Imbalanced | Random  Sampling   | Imbalanced     | Imbalanced   | Imbalanced    |
| classifer2  | Imbalanced | Weighted  Sampling | balanced       | balanced     | Imbalanced    |
| classifier3 | Augmented  | Random  Sampling   | balanced       | balanced     | Imbalanced    |


|              |      | precision | recall | f1     | support |
| ------------ | ---- | --------- | ------ | ------ | ------- |
| classifier 1 | 0    | 0.9185    | 0.9501 | 0.9340 | 842     |
|              | 1    | 0.6744    | 0.5506 | 0.6063 | 158     |
| classifier 2 | 0    | 0.9766    | 0.6948 | 0.8119 | 842     |
|              | 1    | 0.3591    | 0.9114 | 0.5152 | 158     |
| classifier 3 | 0    | 0.9309    | 0.9442 | 0.9375 | 842     |
|              | 1    | 0.6781    | 0.6266 | 0.6513 | 158     |

|              | accuracy | cross entropy |
| ------------ | -------- | ------------- |
| classifier 1 | 0.8870   | 0.03183       |
| classifier 2 | 0.7290   | 0.07741       |
| classifier 3 | 0.8940   | 0.03549       |
  
데이터를 생성해서 추가해서 학습한 모델이 가장 좋은 결과를 낳게됨  
minority class에 대해서 precision은 f1 score은 0.045(7.4%) 상승, accuracy는 0.007(0.78%)상승, cross entropy는 증가함.
