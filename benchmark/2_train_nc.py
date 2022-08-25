import pandas as pd
import numpy as np
import pickle
import random


#datasets = ['BindingDB_KI', 'ChEMBL_KI', 'Davis_KI', 'KIBA_KI', 'DTC_GPCR', 'DTC_SSRI']
datasets = ['Davis_KI', 'DTC_SSRI']

for data in datasets:

    c_space = pd.read_csv('./Benchmark/' + data + '/c_space_' + data + '.csv', index_col='Unnamed: 0')
    t_space = pd.read_csv('./Benchmark/' + data + '/t_space_' + data + '.csv', index_col='Unnamed: 0')
    i_space = pd.read_csv('./Benchmark/' + data + '/i_space.csv', index_col='Unnamed: 0')

    data_tr =  pd.read_csv('./Benchmark/' + data + '/' + data + '_train.csv', index_col='Unnamed: 0')
    train_nc = data_tr.iloc[:,0:3]

    # Train C-space
    train_00_smi = data_tr['SMILES'].drop_duplicates()
    tr_C = pd.merge(train_00_smi, c_space, left_on='SMILES', right_on='SMILES')
    # Train T-space
    train_00_uni = data_tr['TARGETS'].drop_duplicates()
    #t_space = t_space.rename(columns={"Unnamed: 0": "TARGETS"})
    tr_T = pd.merge(train_00_uni, t_space, left_on='TARGETS', right_on='TARGETS')

    # k, q sizes
    if data == 'Davis_KI':
        k=25
        q=25
    elif data == 'DTC_SSRI':
        k=250
        q=10
    else:
        k=250
        q=25

        
    for i in range(len(train_nc)):
        
        # Top ranking compounds in the traning space
        c_x_sim = tr_C.loc[:, tr_C.columns == train_nc.loc[i,'SMILES']].sort_values(by=train_nc.loc[i,'SMILES'], ascending=False)[0:k]
        c_x_ids = tr_C[tr_C.index.isin(tr_C.loc[:, tr_C.columns == train_nc.loc[i,'SMILES']].sort_values(by=train_nc.loc[i,'SMILES'], ascending=False)[0:k].index)]['SMILES']
        c_x = pd.merge(c_x_ids, c_x_sim, left_index=True, right_index=True)

        # Top ranking kinases in the traning space
        t_x_sim = t_space.loc[:, t_space.columns == train_nc.loc[i,'TARGETS']].sort_values(by=train_nc.loc[i,'TARGETS'], ascending=False)[0:q]
        t_x_ids = t_space[t_space.index.isin(t_space.loc[:, t_space.columns == train_nc.loc[i,'TARGETS']].sort_values(by=train_nc.loc[i,'TARGETS'], ascending=False)[0:q].index)]['TARGETS']
        t_x = pd.merge(t_x_ids, t_x_sim, left_index=True, right_index=True)
        
        # Find the interaction landscape of extracted conformity subspaces
        conf_region = pd.merge(c_x_ids, i_space, left_on='SMILES', right_on='SMILES')
        conf_region = conf_region.loc[:, conf_region.columns.isin(t_x_ids)]
        conf_region.insert(0, "SMILES", c_x_ids.values)
        conf_region = pd.melt(conf_region, id_vars=['SMILES'])
        conf_region = conf_region.rename(columns={"variable": "TARGETS", "value": "affinity"})
        conf_region = conf_region.dropna()
        conf_region['mean_diff'] = abs(conf_region['affinity'] - conf_region['affinity'].mean())
        conf_region['y_pred_knn'] = conf_region['affinity'].mean()

        # Unite similarities and differences under one df
        tr_nonconf_ = conf_region.merge(c_x, on="SMILES")
        tr_nonconf_ = tr_nonconf_.merge(t_x, on="TARGETS")
  
        # Retrieve distances
        train_nc.loc[i,'affinity_std'] = tr_nonconf_['affinity'].std()
        
        tr_nonconf_.iloc[:,5] = abs(1-tr_nonconf_.iloc[:,5])
        train_nc.loc[i,'sum_dist_c'] = tr_nonconf_.iloc[:,5].sum()
        
        tr_nonconf_.iloc[:,6] = abs(1-tr_nonconf_.iloc[:,6])
        train_nc.loc[i,'sum_dist_t'] = tr_nonconf_.iloc[:,6].sum()

    train_nc.loc[:, 'sum_dist_ct'] = (train_nc.loc[:, 'sum_dist_c'] + train_nc.loc[:, 'sum_dist_t']) / 2
        
    train_nc.to_csv('./Benchmark/' + data + '/train_nc.csv')
