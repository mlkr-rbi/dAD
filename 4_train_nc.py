import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import random
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from scoring_metrics import rmse, mse, ci
from math import exp, log

c_space = pd.read_csv('./scenarios/c_space.csv')
t_space = pd.read_csv('./scenarios/t_space.csv')
i_space = pd.read_csv('./scenarios/i_space.csv')

train_00 = pd.read_csv('./scenarios/train_00.csv')

train_nc = train_00.iloc[:,0:3]
#ts1_nc_['y_pred'] = model.predict(X_ts1)

# Train C-space
train_00_smi = train_00['SMILES'].drop_duplicates()
tr_C = pd.merge(train_00_smi, c_space, left_on='SMILES', right_on='SMILES')
# Train T-space
train_00_uni = train_00['Uniprot'].drop_duplicates()
tr_T = pd.merge(train_00_uni, t_space, left_on='Uniprot', right_on='Uniprot')


c_mean_sims = []
t_mean_sims = []
tr_conf_sizes = []

for i in range(len(train_nc)):
    
    # Top ranking compounds in the traning space
    c_500_sim = tr_C.loc[:, tr_C.columns == train_nc.loc[i,'SMILES']].sort_values(by=train_nc.loc[i,'SMILES'], ascending=False)[0:250]
    c_500_ids = tr_C[tr_C.index.isin(tr_C.loc[:, tr_C.columns == train_nc.loc[i,'SMILES']].sort_values(by=train_nc.loc[i,'SMILES'], ascending=False)[0:250].index)]['SMILES']
    c_500 = pd.merge(c_500_ids, c_500_sim, left_index=True, right_index=True)

    # Top ranking kinases in the traning space
    t_50_sim = t_space.loc[:, t_space.columns == train_nc.loc[i,'Uniprot']].sort_values(by=train_nc.iloc[i,1], ascending=False)[0:25]
    t_50_ids = t_space[t_space.index.isin(t_space.loc[:, t_space.columns == train_nc.loc[i,'Uniprot']].sort_values(by=train_nc.loc[i,'Uniprot'], ascending=False)[0:25].index)]['Uniprot']
    t_50 = pd.merge(t_50_ids, t_50_sim, left_index=True, right_index=True)
    
    # Find the interaction landscape of extracted conformity subspaces
    conf_region = pd.merge(c_500_ids, i_space, left_on='SMILES', right_on='SMILES')
    conf_region = conf_region.loc[:, conf_region.columns.isin(t_50_ids)]
    conf_region.insert(0, "SMILES", c_500_ids.values)
    conf_region = pd.melt(conf_region, id_vars=['SMILES'])
    conf_region = conf_region.rename(columns={"variable": "Uniprot", "value": "affinity"})
    conf_region = conf_region.dropna()
    conf_region['mean_diff'] = abs(conf_region['affinity'] - conf_region['affinity'].mean())
    #conf_region['pred_diff'] = abs(conf_region['affinity'] - ts1_nc_.loc[i,'y_pred'])
    conf_region['y_pred_knn'] = conf_region['affinity'].mean()

    # Unite similarities and differences under one df
    tr_nonconf_ = conf_region.merge(c_500, on="SMILES")
    tr_nonconf_ = tr_nonconf_.merge(t_50, on="Uniprot")
    #ts1_nonconf["ct_sim"] = (ts1_nonconf.iloc[:,5]/c_norm_const) * (ts1_nonconf.iloc[:,6]/t_norm_const)
    
    c_mean_sims.append(tr_nonconf_.iloc[:,5].mean())
    t_mean_sims.append(tr_nonconf_.iloc[:,6].mean())
    
    # Check conformity region sizes!
    tr_conf_sizes.append(len(tr_nonconf_))
    
    #for j in range(len(tr_nonconf_)):
    #    tr_nonconf_.loc[j,'std'] = tr_nonconf_['affinity'].std()

    #train_nc.loc[i,'median_std'] = tr_nonconf_['std'].median()
    train_nc.loc[i,'affinity_std'] = tr_nonconf_['affinity'].std()
    
    tr_nonconf_.iloc[:,5] = abs(1-tr_nonconf_.iloc[:,5])
    train_nc.loc[i,'sum_dist_c'] = tr_nonconf_.iloc[:,5].sum()
    
    tr_nonconf_.iloc[:,6] = abs(1-tr_nonconf_.iloc[:,6])
    train_nc.loc[i,'sum_dist_t'] = tr_nonconf_.iloc[:,6].sum()
    
train_nc.loc[:, 'sum_dist_ct'] = (train_nc.loc[:, 'sum_dist_c'] + train_nc.loc[:, 'sum_dist_t']) / 2

train_nc.to_csv('./scenarios/train_nc.csv')

