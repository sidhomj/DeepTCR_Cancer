import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df_label = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
df_label = pd.melt(df_label)
df_label.dropna(inplace=True)
df_label.rename(columns={'value':'TCR clonotype family'},inplace=True)
df_label = df_label[df_label['variable']!='Tumor/control reactive']
df_label.rename(columns={'variable':'label'},inplace=True)
df_label = df_label[['TCR clonotype family','label']]

df_merge = pd.merge(df_scores,df_label,on='TCR clonotype family')
df_merge.drop_duplicates(subset=['TCR clonotype family','CDR3B_1','TRBV_1','TRBD_1','TRBJ_1'],inplace=True)

beta_sequences = np.array(df_merge['CDR3B_1'])
v_beta = np.array(df_merge['TRBV_1'])
d_beta = np.array(df_merge['TRBD_1'])
j_beta = np.array(df_merge['TRBJ_1'])
class_labels = np.array(df_merge['label'])

DTCR = DeepTCR_SS('tumor_s')
DTCR.Load_Data(beta_sequences=beta_sequences,v_beta=v_beta,j_beta=j_beta,class_labels=class_labels)
# DTCR.Load_Data(beta_sequences=beta_sequences,class_labels=class_labels)
DTCR.Monte_Carlo_CrossVal(folds=10)
DTCR.Representative_Sequences(make_seq_logos=False,top_seq=50)
DTCR.AUC_Curve()
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(DTCR.Rep_Seq['MAA']['beta'])[0:10],
                              class_sel='MAA')
