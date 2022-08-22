import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define test scenarios
SX = ['S1', 'S2', 'S3', 'S4']

# Load training data
train_00 = pd.read_csv('/scenarios/train_00.csv', index_col='Unnamed: 0')

# Create compound similarity matrix
c_space = pd.read_csv('/home/dorsolic/notebooks/PDB/DATA/CP_2022/data/c_space.csv', index_col='Unnamed: 0')
t_space = pd.read_csv('/home/dorsolic/notebooks/PDB/DATA/CP_2022/data/t_space.csv', index_col='Unnamed: 0')

# Create interaction matrix
i_space = train_00.iloc[:, 0:3]
i_space = i_space.pivot(index='SMILES', columns='Uniprot', values='affinity')
i_space = i_space.reset_index()
i_space.to_csv('/home/dorsolic/notebooks/PDB/DATA/CP_2022/data/i_space.csv', index=False)

# Retain compound similarities toward the training samples
c_space = c_space[c_space['SMILES'].isin(train_00['SMILES'])]
t_space = t_space[t_space['Uniprot'].isin(train_00['Uniprot'])]

for s in SX:

    # UPLOAD DATA
    test_x = pd.read_csv('./scenarios/test_' + s + '.csv', index_col='Unnamed: 0')

    # Map c-similarities toward traning samples
    tr_C_smiles = c_space['SMILES']
    ts_C = c_space.loc[:, c_space.columns.isin(test_x['SMILES'])]
    ts_C.insert(0, "SMILES", tr_C_smiles)
    ts_C.to_csv('./scenarios/' + s + '_C.csv', index=False)

    # Map t-similarities toward traning samples
    tr_T_ids = t_space_['Uniprot']
    ts_T = t_space.loc[:, t_space.columns.isin(test_x['Uniprot'])]
    ts_T.insert(0, "Uniprot", tr_T_ids)
    ts_T.to_csv('./scenarios/' + s + '_T.csv', index=False)
