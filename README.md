# Dynamic Applicability Domain (dAD)
Dynamic applicability domain (dAD) is a extension of conformal predictor framework for approximation of prediction regions with confidence guarantees for dyadic data. We show the performance of the dAD algorithm for compound-target binding affinity space.

## 1. SCKBA
dAD approach applied over SCKBA (small compound-kinase binding affinity dataset) datasest, and tested over four difficulty scenarios (S1-S4).

### 1.1. Download datasets 
Download .zip file with datasets to the root of the repo from `https://drive.google.com/drive/folders/1p-dWKNgbMWXv2WZ6bbdsRgYGOCNwGIFA?usp=sharing`

Datasets include:
- training set (SCKBA)
- test sets (S1-S4)
- compound similarity matrix (Tanimoto)
- target similarity matrix (SW)


### 1.2. Data processing
Create similarity matrices of test compounds and targets towards the training samples.

```
python data_processing.py  
```

### 1.3. Training
Train XGBoost model on the training set; train an additional model in 10x10-fold CV mode to compute nonconformity scores of all training samples.

```
python train_xgb.py 
python train_xgb_cv.py
```


### 1.4. Dynamic applicability domain 

Run a dAD method - required inputs include:
- test dataset(s)
- compound similarities towards the training compounds
- target similarities towards the training targets
- interaction matrix 
- pretrained model

```
python dAD.py
```

