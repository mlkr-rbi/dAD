# dAD
Dynamic applicability domain (dAD) is a extension of conformal predictor framework for approximation of prediction regions with confidence guarantees for dyadic data.

## 1. Download datasets 
Download dataset to the root of the repo from https://drive.google.com/drive/folders/1p-dWKNgbMWXv2WZ6bbdsRgYGOCNwGIFA?usp=sharing

To apply dAD method over small compound kinase inhibitor dataset (SCKBA) run:

## 2. Data processing
```
python data_processing.py  
```
Create similarity matrices of test compounds and targets towards the training samples.

## 3. Training
```
python train_xgb.py 
python train_xgb_cv.py
```

Train XGBoost model on the training set; train an additional model in 10x10-fold CV mode to compute nonconformity scores of all traning samples.

## 4. Dynamic applicability domain 

Run a dAD method - required inputs include:
- test dataset
- compound similarities towards the training compounds
- target similarities towards the training targets
- interaction matrix 
- pretrained model
```
python dAD.py
```

