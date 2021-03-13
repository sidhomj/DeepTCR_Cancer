import pandas as pd
import numpy as np

df = pd.read_csv('CM038_BM.csv')
df_preds = pd.read_csv('sample_tcr_hla.csv')
df_preds['Samples'] = df_preds['Samples'].str[:-9]
df_preds_gp = df_preds.groupby(['Samples']).agg({'y_pred':'mean'}).reset_index()
label_dict = dict(zip(df_preds_gp['Samples'],df_preds_gp['y_pred']))

df['DeepTCR'] = df['sample'].map(label_dict)
df.to_csv('CM038_BM2.csv',index=False)