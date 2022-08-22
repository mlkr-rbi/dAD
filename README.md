# dAD
Dynamic applicability domain (dAD) is a extension of conformal predictor framework for approximation of prediction regions with confidence guarantees for dyadic data.

Step 1. Download datasets from  https://drive.google.com/drive/folders/1p-dWKNgbMWXv2WZ6bbdsRgYGOCNwGIFA?usp=sharing

To apply dAD method over small compound kinase inhibitor dataset (SCKBA) run:

Step 2. data_processing.py  
Create similarity matrices of test compounds and targets towards the training samples.


Step 3. train_xgb.py; train_xgb_cv.py
Train XGBoost model on the training set; train an additional model in 10x10-fold CV mode to compute nonconformity scores of all traning samples.

Step 4. dAD.py
Run a dAD method - required inputs include:
	a) test dataset, 
	b) compound similarities towards the training compounds, 
	c) target similarities towards the training targets, 
	d) interaction matrix and e) pretrained model
