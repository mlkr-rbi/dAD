import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

datasets = ['BindingDB_KI', 'ChEMBL_KI', 'Davis_KI', 'KIBA_KI', 'DTC_GPCR', 'DTC_SSRI']

for data in datasets:

    # UPLOAD DATA
    x_data = pd.read_csv('./Benchmark/' + data + '/' + data + '_dataset.csv', index_col="Unnamed: 0")
    x_data = x_data.drop_duplicates()

    # Split data on train and test
    x_train, x_test = train_test_split(x_data, test_size=0.2, random_state=239)

    c_space =  pd.read_csv('./Benchmark/' + data + '/c_space_' + data + '.csv', index_col='Unnamed: 0')
    t_space =  pd.read_csv('./Benchmark/' + data + '/t_space_' + data + '.csv', index_col='Unnamed: 0')

    # PREPARE C-SPACE, T-SPACE & I-SPACE
    # Map similarities - TRAIN
    tr_C = x_train.merge(c_space, on=['SMILES'])
    tr_T = x_train.merge(t_space, on=['TARGETS'])
    data_tr = tr_T.merge(tr_C, on=['SMILES', 'TARGETS', 'affinity'])
    data_tr.to_csv('./Benchmark/' + data + '/' + data + '_train.csv', index=False)

    # Map similarities - TEST
    ts_C = x_test.merge(c_space, on=['SMILES'])
    ts_T = x_test.merge(t_space, on=['TARGETS'])
    data_ts = ts_T.merge(ts_C, on=['SMILES', 'TARGETS', 'affinity'])
    data_ts.to_csv('./Benchmark/' + data + '/' + data + '_test.csv', index=False)

    # EXTRACT C-SIMILARITY SPACE FROM THE LOOKUP TABLE
    c_space_ = c_space[c_space['SMILES'].isin(data_tr['SMILES'])]

    tr_C_smiles = c_space_['SMILES']
    ts_C_ = c_space_.loc[:, c_space_.columns.isin(data_ts['SMILES'])]
    ts_C_.insert(0, "SMILES", tr_C_smiles)
    ts_C_.to_csv('./Benchmark/' + data + '/ts_C_space.csv', index=False)

    # EXTRACT T-SIMILARITY SPACE FROM THE LOOKUP TABLE (TS1 DATA)
    t_space_ = t_space[t_space['TARGETS'].isin(data_tr['TARGETS'])]
    tr_T_ids = t_space_['TARGETS']
    ts_T_ = t_space_.loc[:, t_space_.columns.isin(data_ts['TARGETS'])]
    ts_T_.insert(0, "TARGETS", tr_T_ids)
    ts_T_.to_csv('./Benchmark/' + data + '/ts_T_space.csv', index=False)

    # Create interaction matrix
    i_space = x_train.iloc[:, 0:3]
    i_space = i_space.drop_duplicates(subset=['SMILES', 'TARGETS'])
    i_space = i_space.pivot(index='SMILES', columns='TARGETS', values='affinity')
    i_space = i_space.reset_index()
    i_space.to_csv('./Benchmark/' + data + '/i_space.csv', index=False)
