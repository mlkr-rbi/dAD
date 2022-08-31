# Dynamic Applicability Domain (dAD)
Dynamic applicability domain (dAD) is a extension of conformal predictor framework for approximation of prediction regions with confidence guarantees for dyadic data. We show the performance of the dAD algorithm for compound-target binding affinity space.

## Requirements
```
pandas >= '1.1.5'
numpy >= '1.19.5'
xgboost >= '1.1.1'
scikit-learn >= '0.22.1'
nonconformist >= '2.1.0'
```

## 1. SCKBA
dAD approach applied over small compound-kinase binding affinity dataset (SCKBA) datasest, and tested over four difficulty scenarios (S1-S4).

### 1.1. Download datasets 
Download .zip file with datasets to the root of the repo from `https://drive.google.com/file/d/1ZTxLLd3-5WToYnIodjJic6aey2FxY7Ho/view?usp=sharing`

Or download directly from command line using gdown:
```
pip install gdown
gdown 1ZTxLLd3-5WToYnIodjJic6aey2FxY7Ho
unzip sckba.zip
```

Datasets include:
- training set (SCKBA)
- test sets (S1-S4)
- compound similarity matrix (Tanimoto)
- target similarity matrix (SW)


### 1.2. Data processing
Create similarity matrices of test compounds and targets towards the training samples.

```
python 1_data_processing.py  
```

### 1.3. Training
Train XGBoost model on the training set; train an additional model in 10x10-fold CV mode to compute nonconformity scores of all training samples.

```
python 1_train_xgb.py 
python 1_train_xgb_cv.py
```


### 1.4. Dynamic applicability domain 

Run a dAD method - required inputs include:
- test dataset(s)
- compound similarities towards the training compounds
- target similarities towards the training targets
- interaction matrix 
- pretrained model

```
python 1_dAD.py
```

## 2. SCKBA (baseline CP)

### 2.1. Training
Using the `nonconformist` library train XGBoost model on the training set and compute calibration scores.

```
python 2_CP_baseline.py 
```

### 2.2. Normalisation coefficients
To compare the dAD approach with baseline studies, we need to compute normalisation coefficients as used in earlier studies.

`2_train_nc.py` computes normasation coefficients regarding the median distance and standard deviations from the training samples
`2_train_xgb_err.py` builds an additional error model, which prediction of error is used as normalisation of nonconformity scores


## 3. BENCHMARK
Same as for the SCKBA dataset, dAD approach (and baseline approaches) could be tested over several benchmark datasets available at `https://drive.google.com/file/d/1yS8p-g_z9Tf6ucw6ey-AQnD_Et8e43tz/view?usp=sharing`, or:

```
gdown 1yS8p-g_z9Tf6ucw6ey-AQnD_Et8e43tz
unzip benchmark.zip
```
Data processing and the rest of the scripts is defined for all six datasets that we used as a benchmark for dAD testing. Depending on the dataset of interest, comment others in the predefined list at the begining of every script.
