import pandas as pd
import numpy as np

df = pd.read_csv('../../Data/other/CM038_BM2.csv')
df_preds = pd.read_csv('../models/sample_tcr_hla.csv')
df_preds['Samples'] = df_preds['Samples'].str[:-9]
df_preds_gp = df_preds.groupby(['Samples']).agg({'y_pred':'mean'}).reset_index()
label_dict = dict(zip(df_preds_gp['Samples'],df_preds_gp['y_pred']))

df['DeepTCR1'] = df['sample'].map(label_dict)
df.to_csv('../../Data/other/CM038_BM2.csv',index=False)