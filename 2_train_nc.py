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

c_space = pd.read_csv('./sckba/c_space.csv')
t_space = pd.read_csv('./sckba/t_space.csv')
i_space = pd.read_csv('./sckba/i_space.csv')

train_00 = pd.read_csv('./sckba/train_00.csv')

train_nc = train_00.iloc[:,0:3]

# Train C-space
train_00_smi = train_00['SMILES'].drop_duplicates()
tr_C = pd.merge(train_00_smi, c_space, left_on='SMILES', right_on='SMILES')

# Train T-space
train_00_uni = train_00['Uniprot'].drop_duplicates()
tr_T = pd.merge(train_00_uni, t_space, left_on='Uniprot', right_on='Uniprot')


c_mean_sims = []
t_mean_sims = []
tr_conf_sizes = []

k=250; q=25

for i in range(len(train_nc)):
    
    # Top ranking compounds in the training space
    c_k_sim = tr_C.loc[:, tr_C.columns == train_nc.loc[i,'SMILES']].sort_values(by=train_nc.loc[i,'SMILES'], ascending=False)[0:k]
    c_k_ids = tr_C[tr_C.index.isin(tr_C.loc[:, tr_C.columns == train_nc.loc[i,'SMILES']].sort_values(by=train_nc.loc[i,'SMILES'], ascending=False)[0:k].index)]['SMILES']
    c_k = pd.merge(c_k_ids, c_k_sim, left_index=True, right_index=True)

    # Top ranking protein targets in the training space
    t_q_sim = t_space.loc[:, t_space.columns == train_nc.loc[i,'Uniprot']].sort_values(by=train_nc.iloc[i,1], ascending=False)[0:q]
    t_q_ids = t_space[t_space.index.isin(t_space.loc[:, t_space.columns == train_nc.loc[i,'Uniprot']].sort_values(by=train_nc.loc[i,'Uniprot'], ascending=False)[0:q].index)]['Uniprot']
    t_q = pd.merge(t_q_ids, t_q_sim, left_index=True, right_index=True)
    
    # Find the binding affinities for the dynamic calibration set
    conf_region = pd.merge(c_k_ids, i_space, left_on='SMILES', right_on='SMILES')
    conf_region = conf_region.loc[:, conf_region.columns.isin(t_q_ids)]
    conf_region.insert(0, "SMILES", c_k_ids.values)
    conf_region = pd.melt(conf_region, id_vars=['SMILES'])
    conf_region = conf_region.rename(columns={"variable": "Uniprot", "value": "affinity"})
    conf_region = conf_region.dropna()
    conf_region['mean_diff'] = abs(conf_region['affinity'] - conf_region['affinity'].mean())
    conf_region['y_pred_knn'] = conf_region['affinity'].mean()

    # Unite similarities and differences under one df
    tr_nonconf_ = conf_region.merge(c_k, on="SMILES")
    tr_nonconf_ = tr_nonconf_.merge(t_q, on="Uniprot")
    
    # Extract mean similarities
    c_mean_sims.append(tr_nonconf_.iloc[:,5].mean())
    t_mean_sims.append(tr_nonconf_.iloc[:,6].mean())
    
    # Check conformity region sizes!
    tr_conf_sizes.append(len(tr_nonconf_))

    # Calculate coefficients
    train_nc.loc[i,'affinity_std'] = tr_nonconf_['affinity'].std()
    
    # lambda [Papadopoulos et al. (9)]
    tr_nonconf_.iloc[:,5] = abs(1-tr_nonconf_.iloc[:,5])
    train_nc.loc[i,'sum_dist_c'] = tr_nonconf_.iloc[:,5].sum()
    
    # ksi [Papadopoulos et al. (10)]
    tr_nonconf_.iloc[:,6] = abs(1-tr_nonconf_.iloc[:,6])
    train_nc.loc[i,'sum_dist_t'] = tr_nonconf_.iloc[:,6].sum()
    

train_nc.loc[:, 'sum_dist_ct'] = (train_nc.loc[:, 'sum_dist_c'] + train_nc.loc[:, 'sum_dist_t']) / 2

# Write a file
train_nc.to_csv('./sckba/train_nc.csv')

