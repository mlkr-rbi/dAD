import pandas as pd
import numpy as np
import pickle
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
import random 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from math import exp, log

# Load common files
train_00 = pd.read_csv('./sckba/train_00.csv')
i_space = pd.read_csv('./sckba/i_space.csv')
train_nc = pd.read_csv('./sckba/train_nc.csv')
model = pickle.load(open('./sckba/output/xg_reg_ERR_base.pkl','rb'))

# Subset data on proper traning and calibration set
SEED=239
random.seed(SEED)
np.random.seed(SEED)
calib = train_00.sample(n=4000, random_state=SEED)
train_00 = train_00.drop(calib.index)

# Prepare data (Training)
X_tr, y_tr = train_00.iloc[:,3:], train_00.iloc[:,2]
X_tr.columns = X_tr.columns.str.replace('[','')
X_tr.columns = X_tr.columns.str.replace(']','')
X_tr.columns = X_tr.columns.str.replace('<','')

# Prepare data (Calibration)
X_cal, y_cal = calib.iloc[:,3:], calib.iloc[:,2]
X_cal.columns = X_cal.columns.str.replace('[','')
X_cal.columns = X_cal.columns.str.replace(']','')
X_cal.columns = X_cal.columns.str.replace('<','')


# TRAIN AND CALIBRATE
model_xgb = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 10, alpha = 10, n_estimators = 500, n_jobs=-1, random_state=SEED)
nc = NcFactory.create_nc(model_xgb)
icp = IcpRegressor(nc)

# Fit the data and calibrate
icp.fit(X_tr.to_numpy(), y_tr.to_numpy())
icp.calibrate(X_cal.to_numpy(), y_cal.to_numpy())



# Set specific files into lists
sx = ['S1', 'S2', 'S3', 'S4']

k = 250; q = 25
gamma = [0,0.2]

# Define a for-loop
for x in range(len(sx)):
    ts_C = pd.read_csv('~/notebooks/PDB/DATA/CP_2022/data/' + sx[x] + "_C.csv")
    ts_T = pd.read_csv('~/notebooks/PDB/DATA/CP_2022/data/' + sx[x] + "_T.csv")

    data_ts = pd.read_csv('./sckba/test_' + s + '.csv')

    X_ts, y_ts = data_ts.iloc[:,3:], data_ts.iloc[:,2]
    X_ts.columns = X_ts.columns.str.replace('[','')
    X_ts.columns = X_ts.columns.str.replace(']','')
    X_ts.columns = X_ts.columns.str.replace('<','')

    # Predict
    ts_prediction_75 = icp.predict(X_ts.to_numpy(), significance=0.25)
    ts_prediction_80 = icp.predict(X_ts.to_numpy(), significance=0.20)
    ts_prediction_85 = icp.predict(X_ts.to_numpy(), significance=0.15)
    ts_prediction_90 = icp.predict(X_ts.to_numpy(), significance=0.10)
    ts_prediction_95 = icp.predict(X_ts.to_numpy(), significance=0.05)
    ts_prediction_99 = icp.predict(X_ts.to_numpy(), significance=0.01)
    
    # Predict error
    ts_ERR = model.predict(X_ts)

    ts_nc = data_ts.iloc[:,0:3]
        
    for i in range(len(ts_nc)):
        
        # Top ranking compounds in the traning space
        c_k_sim = ts_C.loc[:, ts_C.columns == ts_nc.loc[i,'SMILES']].sort_values(by=ts_nc.loc[i,'SMILES'], ascending=False)[0:k]
        c_k_ids = ts_C[ts_C.index.isin(ts_C.loc[:, ts_C.columns == ts_nc.loc[i,'SMILES']].sort_values(by=ts_nc.loc[i,'SMILES'], ascending=False)[0:k].index)]['SMILES']
        c_k = pd.merge(c_k_ids, c_k_sim, left_index=True, right_index=True)

        # Top ranking kinases in the traning space
        t_q_sim = ts_T.loc[:, ts_T.columns == ts_nc.loc[i,'Uniprot']].sort_values(by=ts_nc.loc[i,'Uniprot'], ascending=False)[0:q]
        t_q_ids = ts_T[ts_T.index.isin(ts_T.loc[:, ts_T.columns == ts_nc.loc[i,'Uniprot']].sort_values(by=ts_nc.loc[i,'Uniprot'], ascending=False)[0:q].index)]['Uniprot']
        t_q = pd.merge(t_q_ids, t_q_sim, left_index=True, right_index=True)
        
        # Find the interaction landscape of extracted conformity subspaces
        conf_region = pd.merge(c_k_ids, i_space, left_on='SMILES', right_on='SMILES')
        conf_region = conf_region.loc[:, conf_region.columns.isin(t_q_ids)]
        conf_region.insert(0, "SMILES", c_k_ids.values)
        conf_region = pd.melt(conf_region, id_vars=['SMILES'])
        conf_region = conf_region.rename(columns={"variable": "Uniprot", "value": "affinity"})
        conf_region = conf_region.dropna()

        # Unite similarities and differences under one df
        ts_nonconf_ = conf_region.merge(c_k, on="SMILES")
        ts_nonconf_ = ts_nonconf_.merge(t_q, on="Uniprot")
        
        ts_sim_sum = (ts_nonconf_.iloc[:,3] + ts_nonconf_.iloc[:,4])/2
        lambda_d = sum(ts_sim_sum) / train_nc['sum_dist_ct'].median()
        ksi_s = ts_nonconf_['affinity'].std() / train_nc['affinity_std'].median()
        
        
        gamma = 0
        if scenarios[x] == 'test_S1_00':
            gamma=0.2
        else:
            gamma=0
        
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
        
        ts_nc.loc[i, '75_CI_std'] = (abs(ts_prediction_75[i][0] - ts_prediction_75[i][1])/2)/ ksi_s
        ts_nc.loc[i, '80_CI_std'] = (abs(ts_prediction_80[i][0] - ts_prediction_80[i][1])/2)/ ksi_s
        ts_nc.loc[i, '85_CI_std'] = (abs(ts_prediction_85[i][0] - ts_prediction_85[i][1])/2)/ ksi_s
        ts_nc.loc[i, '90_CI_std'] = (abs(ts_prediction_90[i][0] - ts_prediction_90[i][1])/2)/ ksi_s
        ts_nc.loc[i, '95_CI_std'] = (abs(ts_prediction_95[i][0] - ts_prediction_95[i][1])/2)/ ksi_s
        ts_nc.loc[i, '99_CI_std'] = (abs(ts_prediction_99[i][0] - ts_prediction_99[i][1])/2)/ ksi_s
        

    ts_nc.to_csv('./output/ts_nonconf_' + sx[x] + '_base.csv')
