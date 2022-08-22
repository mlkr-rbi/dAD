import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import pandas as pd
import numpy as np

# LOAD DATA
train_00 = pd.read_csv('~/notebooks/PDB/DATA/CP_2022/data/train_00.csv')

# XGBOOST
X_train, y_train =  train_00.iloc[:,3:], train_00.iloc[:,2]

# A) EDIT COLNAMES - TRAIN
X_train.columns = X_train.columns.str.replace('[','')
X_train.columns = X_train.columns.str.replace(']','')
X_train.columns = X_train.columns.str.replace('<','')


# TRAIN MODEL (10x10-fold CV)
cvx = KFold(n_splits=10, shuffle=True, random_state=239)

model = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=239)

cv_preds = []
for i in range(0,10):
    cv_preds.append(cross_val_predict(model, np.asarray(X_train), np.asarray(y_train), cv=cvx, method='predict', n_jobs=40, verbose=1))

# SAVE FILE
cv_predictions = pd.DataFrame(cv_preds).T
cv_predictions.to_csv('./scenarios/output/cv_predictions.csv')
