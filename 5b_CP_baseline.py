import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from scoring_metrics import rmse, mse, ci
import random 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from math import exp, log


#datasets = ['BindingDB_KI', 'ChEMBL_KI', 'Davis_KI', 'KIBA_KI', 'DTC_GPCR', 'DTC_SSRI']
datasets = ['Davis_KI', 'DTC_SSRI']

for data in datasets:
    
    # Load common files
    data_tr =  pd.read_csv('./Benchmark/' + data + '/' + data + '_train.csv', index_col='Unnamed: 0')
    data_ts =  pd.read_csv('./Benchmark/' + data + '/' + data + '_test.csv', index_col='Unnamed: 0')
    ts_C =  pd.read_csv('./Benchmark/' + data + '/ts_C_space.csv', index_col='Unnamed: 0')
    ts_T =  pd.read_csv('./Benchmark/' + data + '/ts_T_space.csv', index_col='Unnamed: 0')
    i_space = pd.read_csv('./Benchmark/' + data + '/i_space.csv', index_col='Unnamed: 0')
    train_nc = pd.read_csv('./Benchmark/' + data + '/train_nc.csv')
    model = pickle.load(open('./Benchmark/' + data + '/xgb_ERR_model_(base).pkl','rb'))

    # Subset data on proper traning and calibration set
    SEED=239
    random.seed(SEED)
    np.random.seed(SEED)
    calib = data_tr.sample(n=4000, random_state=SEED)
    data_tr = data_tr.drop(calib.index)

    # Prepare data (Training)
    X_tr, y_tr = data_tr.iloc[:,3:], data_tr.iloc[:,2]
    X_tr.columns = X_tr.columns.str.replace('[','')
    X_tr.columns = X_tr.columns.str.replace(']','')
    X_tr.columns = X_tr.columns.str.replace('<','')

    # Prepare data (Calibration)
    X_cal, y_cal = calib.iloc[:,3:], calib.iloc[:,2]
    X_cal.columns = X_cal.columns.str.replace('[','')
    X_cal.columns = X_cal.columns.str.replace(']','')
    X_cal.columns = X_cal.columns.str.replace('<','')
    
    # Prepare data (Test)
    X_ts, y_ts = data_ts.iloc[:,3:], data_ts.iloc[:,2]
    X_ts.columns = X_ts.columns.str.replace('[','')
    X_ts.columns = X_ts.columns.str.replace(']','')
    X_ts.columns = X_ts.columns.str.replace('<','')


    # TRAIN AND CALIBRATE
    model_xgb = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=SEED)
    nc = NcFactory.create_nc(model_xgb)
    icp = IcpRegressor(nc)

    # Fit the data and calibrate
    icp.fit(X_tr.to_numpy(), y_tr.to_numpy())
    icp.calibrate(X_cal.to_numpy(), y_cal.to_numpy())
    
    # Predict with model
    ts_prediction_75 = icp.predict(X_ts.to_numpy(), significance=0.25)
    ts_prediction_80 = icp.predict(X_ts.to_numpy(), significance=0.20)
    ts_prediction_85 = icp.predict(X_ts.to_numpy(), significance=0.15)
    ts_prediction_90 = icp.predict(X_ts.to_numpy(), significance=0.10)
    ts_prediction_95 = icp.predict(X_ts.to_numpy(), significance=0.05)
    ts_prediction_99 = icp.predict(X_ts.to_numpy(), significance=0.01)


    # Predict error
    ts_ERR = model.predict(X_ts)
    
    # Define sensitivity param
    gamma = 0; gamma_ = 0
    if data == 'BindingDB_KI' | data == 'DTC_SSRI':
        gamma=0; gamma_ = 0
    elif data == 'ChEMBL_KI':
        gamma=0.3; gamma_ = 0
    else:
        gamma=0.7; gamma_=0


    # Define nn params (k,q)
    if data == 'Davis_KI':
        k=25; q=25
    elif data == 'DTC_SSRI':
        k=250; q=10
    else:
        k=250; q=25

    # Nonconformity datasets
    ts_nc = data_ts.iloc[:,0:3]
        
    for i in range(len(ts_nc)):
        
        # Top ranking compounds in the traning space
        c_x_sim = ts_C.loc[:, ts_C.columns == ts_nc.loc[i,'SMILES']].sort_values(by=ts_nc.loc[i,'SMILES'], ascending=False)[0:k]
        c_x_ids = ts_C[ts_C.index.isin(ts_C.loc[:, ts_C.columns == ts_nc.loc[i,'SMILES']].sort_values(by=ts_nc.loc[i,'SMILES'], ascending=False)[0:k].index)]['SMILES']
        c_x_ids = pd.merge(c_x_ids, c_x_sim, left_index=True, right_index=True)

        # Top ranking kinases in the traning space
        t_x_sim = ts_T.loc[:, ts_T.columns == ts_nc.loc[i,'TARGETS']].sort_values(by=ts_nc.loc[i,'TARGETS'], ascending=False)[0:q]
        t_x_ids = ts_T[ts_T.index.isin(ts_T.loc[:, ts_T.columns == ts_nc.loc[i,'TARGETS']].sort_values(by=ts_nc.loc[i,'TARGETS'], ascending=False)[0:q].index)]['TARGETS']
        t_x = pd.merge(t_x_ids, t_x_sim, left_index=True, right_index=True)
        
        # Find the interaction landscape of extracted conformity subspaces
        conf_region = pd.merge(c_x_ids, i_space, left_on='SMILES', right_on='SMILES')
        conf_region = conf_region.loc[:, conf_region.columns.isin(t_x_ids)]
        conf_region.insert(0, "SMILES", c_x_ids.values)
        conf_region = pd.melt(conf_region, id_vars=['SMILES'])
        conf_region = conf_region.rename(columns={"variable": "TARGETS", "value": "affinity"})
        conf_region = conf_region.dropna()

        # Unite similarities and differences under one df
        ts_nonconf_ = conf_region.merge(c_x_ids, on="SMILES")
        ts_nonconf_ = ts_nonconf_.merge(t_x, on="TARGETS")
        
        ts_sim_sum = (ts_nonconf_.iloc[:,3] + ts_nonconf_.iloc[:,4])/2
        lambda_d = sum(ts_sim_sum) / train_nc['sum_dist_ct'].median()
        ksi_s = ts_nonconf_['affinity'].std() / train_nc['affinity_std'].median()
        
        
        # B) DECIDE ON PREDICTION INTERVALS
        ts_nc.loc[i, 'y_pred'] = ts_prediction_90[i].mean()
        ts_nc.loc[i, 'abs_err'] = abs(ts_nc['affinity'][i] - ts_nc['y_pred'][i])
        ts_nc.loc[i, '75_CI'] = abs(ts_prediction_75[i][0] - ts_prediction_75[i][1])/2
        ts_nc.loc[i, '80_CI'] = abs(ts_prediction_80[i][0] - ts_prediction_80[i][1])/2
        ts_nc.loc[i, '85_CI'] = abs(ts_prediction_85[i][0] - ts_prediction_85[i][1])/2
        ts_nc.loc[i, '90_CI'] = abs(ts_prediction_90[i][0] - ts_prediction_90[i][1])/2
        ts_nc.loc[i, '95_CI'] = abs(ts_prediction_95[i][0] - ts_prediction_95[i][1])/2
        ts_nc.loc[i, '99_CI'] = abs(ts_prediction_99[i][0] - ts_prediction_99[i][1])/2
        
        ts_nc.loc[i, '75_CI_err'] = (abs(ts_prediction_75[i][0] - ts_prediction_75[i][1])/2) / exp(log(abs(ts_ERR[i])))
        ts_nc.loc[i, '80_CI_err'] = (abs(ts_prediction_80[i][0] - ts_prediction_80[i][1])/2) / exp(log(abs(ts_ERR[i])))
        ts_nc.loc[i, '85_CI_err'] = (abs(ts_prediction_85[i][0] - ts_prediction_85[i][1])/2) / exp(log(abs(ts_ERR[i])))
        ts_nc.loc[i, '90_CI_err'] = (abs(ts_prediction_90[i][0] - ts_prediction_90[i][1])/2) / exp(log(abs(ts_ERR[i])))
        ts_nc.loc[i, '95_CI_err'] = (abs(ts_prediction_95[i][0] - ts_prediction_95[i][1])/2) / exp(log(abs(ts_ERR[i])))
        ts_nc.loc[i, '99_CI_err'] = (abs(ts_prediction_99[i][0] - ts_prediction_99[i][1])/2) / exp(log(abs(ts_ERR[i])))

        ts_nc.loc[i, '75_CI_dist'] = (abs(ts_prediction_75[i][0] - ts_prediction_75[i][1])/2)/ (gamma + lambda_d)
        ts_nc.loc[i, '80_CI_dist'] = (abs(ts_prediction_80[i][0] - ts_prediction_80[i][1])/2)/ (gamma + lambda_d)
        ts_nc.loc[i, '85_CI_dist'] = (abs(ts_prediction_85[i][0] - ts_prediction_85[i][1])/2)/ (gamma + lambda_d)
        ts_nc.loc[i, '90_CI_dist'] = (abs(ts_prediction_90[i][0] - ts_prediction_90[i][1])/2)/ (gamma + lambda_d)
        ts_nc.loc[i, '95_CI_dist'] = (abs(ts_prediction_95[i][0] - ts_prediction_95[i][1])/2)/ (gamma + lambda_d)
        ts_nc.loc[i, '99_CI_dist'] = (abs(ts_prediction_99[i][0] - ts_prediction_99[i][1])/2)/ (gamma + lambda_d)
        
        ts_nc.loc[i, '75_CI_std'] = (abs(ts_prediction_75[i][0] - ts_prediction_75[i][1])/2)/ (gamma_ + ksi_s)
        ts_nc.loc[i, '80_CI_std'] = (abs(ts_prediction_80[i][0] - ts_prediction_80[i][1])/2)/ (gamma_ + ksi_s)
        ts_nc.loc[i, '85_CI_std'] = (abs(ts_prediction_85[i][0] - ts_prediction_85[i][1])/2)/ (gamma_ + ksi_s)
        ts_nc.loc[i, '90_CI_std'] = (abs(ts_prediction_90[i][0] - ts_prediction_90[i][1])/2)/ (gamma_ + ksi_s)
        ts_nc.loc[i, '95_CI_std'] = (abs(ts_prediction_95[i][0] - ts_prediction_95[i][1])/2)/ (gamma_ + ksi_s)
        ts_nc.loc[i, '99_CI_std'] = (abs(ts_prediction_99[i][0] - ts_prediction_99[i][1])/2)/ (gamma_ + ksi_s)
        
        #ts_nc.loc[i, 'gamma'] = gamma


    ts_nc.to_csv('~/Benchmark/' + data +'/output/ts_nonconf_BASE_' + data + '.csv')
