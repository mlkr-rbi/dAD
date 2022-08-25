import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import pandas as pd
import random
import numpy as np

SEED=239
random.seed(SEED)
np.random.seed(SEED)

# UPLOAD DATA
train_00 = pd.read_csv('./sckba/train_00.csv')

# Subset data on proper traning and calibration set
calib = train_00.sample(n=4000, random_state=SEED)
train_00 = train_00.drop(calib.index)
train_00 = train_00.reset_index()
train_00 = train_00.iloc[:,1:]

cv_tr = pd.read_csv('./output/cv_predictions_BASELINE.csv', index_col='Unnamed: 0')
cv_tr_mean = (cv_tr.iloc[:,0] + cv_tr.iloc[:,1] + cv_tr.iloc[:,2] + cv_tr.iloc[:,3] + cv_tr.iloc[:,4] + cv_tr.iloc[:,5] + cv_tr.iloc[:,6] + cv_tr.iloc[:,7] + cv_tr.iloc[:,8] + cv_tr.iloc[:,9]) / 10

# XGBOOST
X_train, y_train = train_00.iloc[:,3:], abs(cv_tr_mean - train_00.iloc[:,2])

# CLEAN COLUMN NAMES
# A) EDIT COLNAMES - TRAIN
X_train.columns = X_train.columns.str.replace('[','')
X_train.columns = X_train.columns.str.replace(']','')
X_train.columns = X_train.columns.str.replace('<','')


# TRAIN MODEL
xg_reg_ERR = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=SEED)
xg_reg_ERR.fit(X_train, y_train)


# SAVE MODEL!
pickle.dump(xg_reg_ERR, open('./output/xg_reg_ERR_baseline.pkl', 'wb'))
