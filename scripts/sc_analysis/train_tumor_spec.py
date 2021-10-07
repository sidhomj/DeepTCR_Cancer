import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS
from DeepTCR.functions.data_processing import supertype_conv_op


df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)
df_scores.dropna(subset=['CDR3A_1'],inplace=True)

df_label = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
df_label = pd.melt(df_label)
df_label.dropna(inplace=True)
df_label.rename(columns={'value':'TCR clonotype family'},inplace=True)
df_label = df_label[df_label['variable']!='Tumor/control reactive']
df_label.rename(columns={'variable':'label'},inplace=True)
df_label = df_label[['TCR clonotype family','label']]

df_merge = pd.merge(df_scores,df_label,on='TCR clonotype family')
df_merge.drop_duplicates(subset=['TCR clonotype family','CDR3B_1','TRBV_1','TRBD_1','TRBJ_1'],inplace=True)
# df_merge = df_merge[df_merge['patient'] == 'p2']

beta_sequences = np.array(df_merge['CDR3B_1'])
v_beta = np.array(df_merge['TRBV_1'])
d_beta = np.array(df_merge['TRBD_1'])
j_beta = np.array(df_merge['TRBJ_1'])
alpha_sequences = np.array(df_merge['CDR3A_1'])
v_alpha = np.array(df_merge['TRAV_1'])
j_alpha = np.array(df_merge['TRAJ_1'])
hla = np.array(df_merge[['0','1','2','3','4','5']])
hla = np.array(supertype_conv_op(hla))
class_labels = np.array(df_merge['label'])

DTCR = DeepTCR_SS('tumor_s')
DTCR.Load_Data(beta_sequences=beta_sequences,v_beta=v_beta,j_beta=j_beta,
               alpha_sequences=alpha_sequences,v_alpha=v_alpha,j_alpha=j_alpha,
               hla=hla,
               class_labels=class_labels)
DTCR.use_hla = False
DTCR.Monte_Carlo_CrossVal(folds=100)
DTCR.Representative_Sequences(make_seq_logos=False,top_seq=50)
DTCR.AUC_Curve()
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(DTCR.Rep_Seq['MAA']['beta'])[0:25],
                              alpha_sequences=np.array(DTCR.Rep_Seq['MAA']['alpha'])[0:25],
                              class_sel='MAA',models=['model_'+str(x) for x in np.random.choice(range(100),10)],
                              background_color='black',
                              figsize=(3,8),
                              Load_Prev_Data=False,
                              min_size=0.5)

