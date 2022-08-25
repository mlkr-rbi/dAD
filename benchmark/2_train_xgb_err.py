import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import pandas as pd
import numpy as np
import random 

datasets = ['BindingDB_KI', 'ChEMBL_KI', 'Davis_KI', 'KIBA_KI', 'DTC_GPCR', 'DTC_SSRI']

for data in datasets:

    SEED=239
    random.seed(SEED)
    np.random.seed(SEED)

    # UPLOAD DATA
    data_tr = pd.read_csv('./Benchmark/' + data + '/' + data + '_train.csv')

    if (data == 'BindingDB_KI') | (data == 'ChEMBL_KI') | (data == 'KIBA_KI'):
        N = 3000
    else:
        N = 1500

    calib = data_tr.sample(n=N, random_state=SEED)
    data_tr = data_tr.drop(calib.index)
    data_tr = data_tr.reset_index()
    data_tr = data_tr.iloc[:,1:]

    # XGBOOST
    X_tr, y_tr =  data_tr.iloc[:,3:], data_tr.iloc[:,2]

    # CLEAN COLUMN NAMES
    # A) EDIT COLNAMES - TRAIN
    X_tr.columns = X_tr.columns.str.replace('[','', regex=True)
    X_tr.columns = X_tr.columns.str.replace(']','', regex=True)
    X_tr.columns = X_tr.columns.str.replace('<','', regex=True)
    
    # TRAIN MODEL (10x10-fold CV)
    cvx = KFold(n_splits=10, shuffle=True, random_state=239)
    model = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=SEED)

    cv_preds = []
    for i in range(0,10):
        cv_preds.append(cross_val_predict(model, np.asarray(X_tr), np.asarray(y_tr), cv=cvx, method='predict', n_jobs=40, verbose=1))


    # TRAIN ERROR MODEL
    cv_tr = pd.DataFrame(cv_preds).T
    cv_tr_mean = (cv_tr.iloc[:,0] + cv_tr.iloc[:,1] + cv_tr.iloc[:,2] + cv_tr.iloc[:,3] + cv_tr.iloc[:,4] + cv_tr.iloc[:,5] + cv_tr.iloc[:,6] + cv_tr.iloc[:,7] + cv_tr.iloc[:,8] + cv_tr.iloc[:,9]) / 10

    if (data == 'BindingDB_KI') | (data == 'ChEMBL_KI') | (data == 'KIBA_KI'):
        N = 3000
    else:
        N = 1500

    # XGBOOST
    X_tr, y_tr =  data_tr.iloc[:,3:], abs(cv_tr_mean - data_tr.iloc[:,3])

    # CLEAN COLUMN NAMES
    # A) EDIT COLNAMES - TRAIN
    X_tr.columns = X_tr.columns.str.replace('[','', regex=True)
    X_tr.columns = X_tr.columns.str.replace(']','', regex=True)
    X_tr.columns = X_tr.columns.str.replace('<','', regex=True)


    # TRAIN MODEL
    xgb_err = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=SEED)
    xgb_err.fit(X_tr, y_tr)

    # SAVE MODEL!
    pickle.dump(xgb_err, open('./Benchmark/' + data + '/xgb_ERR_model_base.pkl', 'wb'))
