import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df_label = pd.read_csv('../../SC/antigen.csv')
df_label = pd.melt(df_label,id_vars=[df_label.columns[0]],
               value_vars=['NeoAg', 'Cathegory','norm TCR activation', 'Avidity'])
df_label = df_label[df_label['variable']=='Cathegory']
df_label= df_label[['TCR clonotype family','value']]
df_label.rename(columns={'value':'label'},inplace=True)
df_label.reset_index(drop=True,inplace=True)
df_label['label'][df_label['label'] == 'Multi'] = 'NeoAg'

df_merge = pd.merge(df_scores,df_label,on='TCR clonotype family')
df_merge.drop_duplicates(subset=['TCR clonotype family','CDR3B_1','TRBV_1','TRBD_1','TRBJ_1'],inplace=True)
# df_merge.drop_duplicates(subset=['TCR clonotype family','CDR3B_1'],inplace=True)

beta_sequences = np.array(df_merge['CDR3B_1'])
v_beta = np.array(df_merge['TRBV_1'])
d_beta = np.array(df_merge['TRBD_1'])
j_beta = np.array(df_merge['TRBJ_1'])
class_labels = np.array(df_merge['label'])

DTCR = DeepTCR_SS('tumor_s')
DTCR.Load_Data(beta_sequences=beta_sequences,v_beta=v_beta,j_beta=j_beta,class_labels=class_labels)
DTCR.Monte_Carlo_CrossVal(folds=100)
DTCR.Representative_Sequences(make_seq_logos=False,top_seq=50)
DTCR.AUC_Curve()
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(DTCR.Rep_Seq['Viral']['beta'])[0:25],
                              class_sel='Viral',models=['model_'+str(x) for x in np.random.choice(range(100),10)],
                              background_color='black',
                              figsize=(3,8),
                              Load_Prev_Data=False)
