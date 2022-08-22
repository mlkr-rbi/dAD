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
train_00 = pd.read_csv('./scenarios/train_00.csv')

# Subset data on proper traning and calibration set
ssampl = train_00.sample(n=4000, random_state=SEED)
sstrain = train_00.drop(ssampl.index)
sstrain = sstrain.reset_index()
sstrain = sstrain.iloc[:,1:]

cv_tr = pd.read_csv('./scenarios/cv_predictions_BASELINE.csv', index_col='Unnamed: 0')
cv_tr_mean = (cv_tr.iloc[:,0] + cv_tr.iloc[:,1] + cv_tr.iloc[:,2] + cv_tr.iloc[:,3] + cv_tr.iloc[:,4] + cv_tr.iloc[:,5] + cv_tr.iloc[:,6] + cv_tr.iloc[:,7] + cv_tr.iloc[:,8] + cv_tr.iloc[:,9]) / 10

# XGBOOST
X_train, y_train = sstrain.iloc[:,3:], abs(cv_tr_mean - sstrain.iloc[:,2])

# CLEAN COLUMN NAMES
# A) EDIT COLNAMES - TRAIN
X_train.columns = X_train.columns.str.replace('[','')
X_train.columns = X_train.columns.str.replace(']','')
X_train.columns = X_train.columns.str.replace('<','')


# TRAIN MODEL
xg_reg_ERR = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=239)
xg_reg_ERR.fit(X_train, y_train)


# SAVE MODEL!
pickle.dump(xg_reg_ERR, open('./scenarios/xg_reg_ERR_baseline.pkl', 'wb'))
