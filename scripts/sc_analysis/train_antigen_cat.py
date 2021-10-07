import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS
def process_gene(gene,chain,df_merge):
    temp = df_merge['TR'+chain+gene+'_1'].str.split('TR'+chain+gene,expand=True)[1].str.split('-',expand=True)
    temp.fillna(value=1,inplace=True)
    temp[0] = pd.to_numeric(temp[0], errors='coerce')
    if temp.shape[1]==2:
        temp[1] = pd.to_numeric(temp[1],errors='coerce')
    temp = temp.dropna().astype(int)
    temp[0] = temp[0].astype(int).map("{:02}".format)
    if temp.shape[1]==2:
        temp[1] = temp[1].astype(int).map("{:02}".format)
    if temp.shape[1]==2:
        temp[2] = 'TCR'+chain + gene + temp[0].astype(str) + '-'+temp[1].astype(str)
        return temp[2]
    else:
        temp[1] = 'TCR'+chain + gene + temp[0].astype(str)
        return temp[1]

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)
df_scores.dropna(subset=['CDR3A_1'],inplace=True)
# df_scores['TRAV_1'] = process_gene('V','A',df_scores)
# df_scores['TRAJ_1'] = process_gene('J','A',df_scores)

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
alpha_sequences = np.array(df_merge['CDR3A_1'])
v_alpha = np.array(df_merge['TRAV_1'])
j_alpha = np.array(df_merge['TRAJ_1'])
class_labels = np.array(df_merge['label'])

DTCR = DeepTCR_SS('tumor_s')
DTCR.Load_Data(beta_sequences=beta_sequences,v_beta=v_beta,j_beta=j_beta,
               alpha_sequences=alpha_sequences,v_alpha=v_alpha,j_alpha=j_alpha,
               class_labels=class_labels)
folds = 100
DTCR.Monte_Carlo_CrossVal(folds=folds)
DTCR.Representative_Sequences(make_seq_logos=False,top_seq=50)
DTCR.AUC_Curve()
class_sel = 'MAA'
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(DTCR.Rep_Seq[class_sel]['beta'])[0:25],
                              alpha_sequences=np.array(DTCR.Rep_Seq[class_sel]['alpha'])[0:25],
                              class_sel=class_sel,models=['model_'+str(x) for x in np.random.choice(range(folds),10)],
                              background_color='black',
                              figsize=(5,8),
                              Load_Prev_Data=False,
                              min_size=0.5)
