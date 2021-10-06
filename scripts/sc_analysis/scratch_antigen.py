import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS


df_scores = pd.read_csv('tcrs_scored.csv')
df_spec = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
df_spec = pd.melt(df_spec)
df_spec.dropna(inplace=True)
df_spec.rename(columns={'value':'TCR clonotype family'},inplace=True)

df_antigen = pd.read_csv('../../SC/antigen.csv')
df_antigen = pd.melt(df_antigen,id_vars=[df_antigen.columns[0]],
               value_vars=['NeoAg', 'Cathegory','norm TCR activation', 'Avidity'])

df_merge = pd.merge(df_spec,df_antigen,on='TCR clonotype family')

df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)
df_merge2 = pd.merge(df_scores,df_merge,on='TCR clonotype family')
sns.violinplot(data=df_merge2,x='variable_y',y='pred',cut=0)

df_plot = df_merge2[df_merge2['variable_x']=='Tumor specific']
df_plot = df_merge2[df_merge2['variable_x']=='Non Tumor reactive']
df_plot = df_merge2
df_plot = df_plot[df_plot['variable_y']=='Avidity']
sns.boxplot(data=df_plot,x='value',y='pred')
import matplotlib.pyplot as plt
plt.scatter(df_plot['pred'],df_plot['value'])


df_analysis = df_merge2.drop_duplicates(subset=['TCR clonotype family','cell.barcode 2','CDR3B_1'])
df_analysis = df_merge2.drop_duplicates(subset=['CDR3B_1'])

beta_sequences = np.array(df_analysis['CDR3B_1'])
v_beta = np.array(df_analysis['TRBV_1'])
d_beta = np.array(df_analysis['TRBD_1'])
j_beta = np.array(df_analysis['TRBJ_1'])
hla = np.array(df_analysis[['0','1','2','3','4','5']])
class_labels = np.array(df_analysis['variable_x'])

DTCR = DeepTCR_SS('tumor_s')
DTCR.Load_Data(beta_sequences=beta_sequences,v_beta=v_beta,j_beta=j_beta,class_labels=class_labels)
DTCR.Monte_Carlo_CrossVal(folds=10)
DTCR.Representative_Sequences(make_seq_logos=False,top_seq=50)