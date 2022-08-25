import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import pandas as pd
import numpy as np
import random 


SEED=239
random.seed(SEED)
np.random.seed(SEED)

# LOAD DATA
train_00 = pd.read_csv('./sckba/train_00.csv')

# Subset data on proper traning and calibration set
calib = train_00.sample(n=4000, random_state=SEED)
train_00 = train_00.drop(calib.index)

# XGBOOST
X_train, y_train =  train_00.iloc[:,3:], train_00.iloc[:,2]

# A) EDIT COLNAMES - TRAIN
X_train.columns = X_train.columns.str.replace('[','')
X_train.columns = X_train.columns.str.replace(']','')
X_train.columns = X_train.columns.str.replace('<','')


# TRAIN MODEL (10x10-fold CV)
cvx = KFold(n_splits=10, shuffle=True, random_state=239)

model = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=SEED)

cv_preds = []
for i in range(0,10):
    cv_preds.append(cross_val_predict(model, np.asarray(X_train), np.asarray(y_train), cv=cvx, method='predict', n_jobs=-1, verbose=1))

# SAVE FILE
cv_predictions = pd.DataFrame(cv_preds).T
cv_predictions.to_csv('./output/cv_predictions_BASELINE.csv')
