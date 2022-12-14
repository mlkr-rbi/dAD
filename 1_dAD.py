import pandas as pd
import numpy as np
import pickle

sx = ['S1', 'S2', 'S3', 'S4']
modes = ['NN', 'CV']

i_space =  pd.read_csv('./sckba/i_space.csv')
model = pickle.load(open('./output/xgb_reg.pkl','rb'))


for s in sx:

    # Load data and model
    data_ts =  pd.read_csv('./sckba/test_' + s + '.csv')
    ts_C =  pd.read_csv('./sckba/' + s + '_C.csv')
    ts_T =  pd.read_csv('./sckba/' + s + '_T.csv')

    # Prepare data for prediction
    X_ts, y_ts = data_ts.iloc[:,3:], data_ts.iloc[:,2]
    X_ts.columns = X_ts.columns.str.replace('[','', regex=True)
    X_ts.columns = X_ts.columns.str.replace(']','', regex=True)
    X_ts.columns = X_ts.columns.str.replace('<','', regex=True)
    
    # k, q sizes
    k = 250; q = 25

    for m in modes:
        if m  == 'NN':

            # Non-conformity set
            ts_nc = data_ts.iloc[:,0:3]
            ts_nc['y_pred'] = model.predict(X_ts)
            ts_nc['abs_err'] = abs(ts_nc['affinity'] - ts_nc['y_pred'])

            c_mean_sims = []
            t_mean_sims = []
            ts_conf_sizes = []
                
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
                conf_region['mean_diff'] = abs(conf_region['affinity'] - conf_region['affinity'].mean())
                conf_region['pred_diff'] = abs(conf_region['affinity'] - ts_nc.loc[i,'y_pred'])
                
                # Unite similarities and differences under one df
                ts_nonconf = conf_region.merge(c_k, on="SMILES")
                ts_nonconf = ts_nonconf.merge(t_q, on="Uniprot")
                
                # Retrieve mean similarities
                c_mean_sims.append(ts_nonconf.iloc[:,5].mean())
                t_mean_sims.append(ts_nonconf.iloc[:,6].mean())
                
                # Check conformity region sizes!
                ts_conf_sizes.append(len(ts_nonconf))
                
                # Compute alpha scores
                for j in range(len(ts_nonconf)):
                    ts_nonconf.loc[j,'alpha_sc'] = (ts_nonconf[ts_nonconf['pred_diff'] <= ts_nonconf.loc[j,'mean_diff']].shape[0]) / ts_nonconf.shape[0]
                
                
                # Locate smallest alpha for any confidence level
                ts_nc.loc[i,'75_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.75,0.79)].mean_diff)
                ts_nc.loc[i,'80_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.8,0.84)].mean_diff)
                ts_nc.loc[i,'85_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.85,0.89)].mean_diff)
                ts_nc.loc[i,'90_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.90,0.94)].mean_diff)
                ts_nc.loc[i,'95_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.95,0.98)].mean_diff)
                ts_nc.loc[i,'99_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.99,1)].mean_diff)


            # Write files
            ts_conf_sizes_df = pd.DataFrame(ts_conf_sizes)
            ts_conf_sizes_df.to_csv('./output/' + s + '_conf_sizes_NN.csv')
            ts_nc.to_csv('./output/' + s + '_nonconf_NN.csv')


        else:
            # Assign CV predictions for training samples
            data_tr =  pd.read_csv('./sckba/train_00.csv')
            cv_tr = pd.read_csv('./output/cv_predictions.csv', index_col='Unnamed: 0')
            cv_tr_mean = (cv_tr.iloc[:,0] + cv_tr.iloc[:,1] + cv_tr.iloc[:,2] + cv_tr.iloc[:,3] + cv_tr.iloc[:,4] + cv_tr.iloc[:,5] + cv_tr.iloc[:,6] + cv_tr.iloc[:,7] + cv_tr.iloc[:,8] + cv_tr.iloc[:,9]) / 10

            data_tr = data_tr.iloc[:,0:3]
            data_tr['cv_pred'] = cv_tr_mean

            ts_nc = data_ts.iloc[:,0:3]
            ts_nc['y_pred'] = model.predict(X_ts)
            ts_nc['abs_err'] = abs(ts_nc['affinity'] - ts_nc['y_pred'])

            c_mean_sims = []
            t_mean_sims = []
            ts_conf_sizes = []

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
                conf_region = pd.merge(conf_region, data_tr, on=["SMILES", "Uniprot", "affinity"])

                conf_region['cv_diff'] = abs(conf_region['affinity'] - conf_region['cv_pred'])
                conf_region['pred_diff'] = abs(conf_region['affinity'] - ts_nc.loc[i,'y_pred'])
                conf_region = conf_region.drop('cv_pred', 1)

                # Unite similarities and differences under one df
                ts_nonconf = conf_region.merge(c_k, on="SMILES")
                ts_nonconf = ts_nonconf.merge(t_q, on="Uniprot")
                
                # Retrieve mean similarities
                c_mean_sims.append(ts_nonconf.iloc[:,5].mean())
                t_mean_sims.append(ts_nonconf.iloc[:,6].mean())
                
                # Check conformity region sizes!
                ts_conf_sizes.append(len(ts_nonconf))
                
                for j in range(len(ts_nonconf)):
                    ts_nonconf.loc[j,'alpha_sc'] = (ts_nonconf[ts_nonconf['pred_diff'] <= ts_nonconf.loc[j,'cv_diff']].shape[0]) / ts_nonconf.shape[0]
                
                
                # Locate smallest alpha for any confidence level
                ts_nc.loc[i,'75_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.75,0.79)].cv_diff)
                ts_nc.loc[i,'80_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.8,0.84)].cv_diff)
                ts_nc.loc[i,'85_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.85,0.89)].cv_diff)
                ts_nc.loc[i,'90_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.90,0.94)].cv_diff)
                ts_nc.loc[i,'95_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.95,0.98)].cv_diff)
                ts_nc.loc[i,'99_CI'] = np.min(ts_nonconf[ts_nonconf['alpha_sc'].between(0.99,1)].cv_diff)

            # Write files
            ts_conf_sizes_df = pd.DataFrame(ts_conf_sizes)
            ts_conf_sizes_df.to_csv('./output/' + s + '_conf_sizes_CV.csv')
            ts_nc.to_csv('./output/' + s + '_nonconf_CV.csv')
