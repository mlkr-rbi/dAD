import xgboost as xgb
import pickle
import pandas as pd


#datasets = ['BindingDB_KI', 'ChEMBL_KI', 'Davis_KI', 'KIBA_KI', 'DTC_GPCR', 'DTC_SSRI']
datasets = ['Davis_KI', 'DTC_SSRI']

for data in datasets:
    # UPLOAD DATA
    data_tr = pd.read_csv('./Benchmark/' + data + '/' + data + '_train.csv')
    data_ts = pd.read_csv('./Benchmark/' + data + '/' + data + '_test.csv')

    # XGBOOST
    X_tr, y_tr =  data_tr.iloc[:,3:], data_tr.iloc[:,2]
    X_ts, y_ts = data_ts.iloc[:,3:], data_ts.iloc[:,2]

    # CLEAN COLUMN NAMES
    # A) EDIT COLNAMES - TRAIN
    X_tr.columns = X_tr.columns.str.replace('[','', regex=True)
    X_tr.columns = X_tr.columns.str.replace(']','', regex=True)
    X_tr.columns = X_tr.columns.str.replace('<','', regex=True)

    # B) EDIT_COLNAMES - TEST
    X_ts.columns = X_ts.columns.str.replace('[','', regex=True)
    X_ts.columns = X_ts.columns.str.replace(']','', regex=True)
    X_ts.columns = X_ts.columns.str.replace('<','', regex=True)


    # TRAIN MODEL
    xgb_reg = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=239)
    xgb_reg.fit(X_tr, y_tr)


    # PREDICT and WRITE
    ts_pred = pd.DataFrame(xgb_reg.predict(X_ts))
    ts_pred.to_csv('./Benchmark/' + data + '/output/ts_predictions.csv')


    # SAVE MODEL!
    pickle.dump(xgb_reg, open('./Benchmark/' + data + '/xgb_model.pkl', 'wb'))
