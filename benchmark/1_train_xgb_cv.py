import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import pandas as pd
import numpy as np

datasets = ['BindingDB_KI', 'ChEMBL_KI', 'Davis_KI', 'KIBA_KI', 'DTC_GPCR', 'DTC_SSRI']

for data in datasets:
    # UPLOAD DATA
    data_tr = pd.read_csv('./Benchmark/' + data + '/' + data + '_train.csv')

    # XGBOOST
    X_tr, y_tr =  data_tr.iloc[:,3:], data_tr.iloc[:,2]

    # CLEAN COLUMN NAMES
    # A) EDIT COLNAMES - TRAIN
    X_tr.columns = X_tr.columns.str.replace('[','', regex=True)
    X_tr.columns = X_tr.columns.str.replace(']','', regex=True)
    X_tr.columns = X_tr.columns.str.replace('<','', regex=True)
    
    # TRAIN MODEL (10x10-fold CV)
    cvx = KFold(n_splits=10, shuffle=True, random_state=239)
    model = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=239)

    cv_preds = []
    for i in range(0,10):
        cv_preds.append(cross_val_predict(model, np.asarray(X_tr), np.asarray(y_tr), cv=cvx, method='predict', n_jobs=40, verbose=1))

    # SAVE FILE
    cv_predictions = pd.DataFrame(cv_preds).T
    cv_predictions.to_csv('./Benchmark/' + data + '/output/cv_predictions.csv')
