import xgboost as xgb
from sklearn.model_selection import cross_val_predict, KFold
import pickle
import pandas as pd


datasets = ['BindingDB_KI', 'ChEMBL_KI', 'Davis_KI', 'KIBA_KI', 'DTC_GPCR', 'DTC_SSRI']

for data in datasets:

    SEED=239
    random.seed(SEED)
    np.random.seed(SEED)

    # UPLOAD DATA
    data_tr = pd.read_csv('./Benchmark/' + data + '/' + data + '_train.csv')
    cv_tr = pd.read_csv('./Benchmark/' + data + '/output/cv_predictions.csv', index_col='Unnamed: 0')
    cv_tr_mean = (cv_tr.iloc[:,0] + cv_tr.iloc[:,1] + cv_tr.iloc[:,2] + cv_tr.iloc[:,3] + cv_tr.iloc[:,4] + cv_tr.iloc[:,5] + cv_tr.iloc[:,6] + cv_tr.iloc[:,7] + cv_tr.iloc[:,8] + cv_tr.iloc[:,9]) / 10

    if (data == 'BindingDB_KI') | (data == 'ChEMBL_KI') | (data == 'KIBA_KI'):
        N = 3000
    else:
        N = 1500

    # XGBOOST
    X_train, y_train = train_00.iloc[:,3:], abs(cv_tr_mean - train_00.iloc[:,2])

    # CLEAN COLUMN NAMES
    # A) EDIT COLNAMES - TRAIN
    X_train.columns = X_train.columns.str.replace('[','', regex=True)
    X_train.columns = X_train.columns.str.replace(']','', regex=True)
    X_train.columns = X_train.columns.str.replace('<','', regex=True)


    # TRAIN MODEL
    xgb_err = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=SEED)
    xgb_err.fit(X_train, y_train)

    # SAVE MODEL!
    pickle.dump(xgb_err, open('./Benchmark/' + data + '/xgb_ERR_model.pkl', 'wb'))
