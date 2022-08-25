import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import pandas as pd

# LOAD DATA
train_00 = pd.read_csv('./sckba/train_00.csv')

# XGBOOST
X_train, y_train =  train_00.iloc[:,3:], train_00.iloc[:,2]

# A) EDIT COLNAMES - TRAIN
X_train.columns = X_train.columns.str.replace('[','')
X_train.columns = X_train.columns.str.replace(']','')
X_train.columns = X_train.columns.str.replace('<','')


# TRAIN MODEL
import xgboost as xgb
xg_reg = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=239)
xg_reg.fit(X_train, y_train)

pickle.dump(xg_reg, open('./output/xgb_reg.pkl', 'wb'))

