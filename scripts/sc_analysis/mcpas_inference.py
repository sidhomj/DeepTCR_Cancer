import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DeepTCR.functions.data_processing import Process_Seq
from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF


df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)
df_scores.dropna(subset=['CDR3A_1'],inplace=True)

df_scores.drop(columns=['cell.barcode'],inplace=True)
df_scores.rename(columns={'cell.barcode 2':'cell.barcode'},inplace=True)

df = pd.read_csv('../../SC/umap.csv')
df.rename(columns={df.columns[0]:'cell.barcode'},inplace=True)
df_merge = pd.merge(df,df_scores,on='cell.barcode')
df_all = df_merge

df = pd.read_csv('McPAS-TCR.csv')
df = df[df['Species']=='Human']
df.dropna(subset=['CDR3.beta.aa','Category'],inplace=True)
df = df[df['Category'].isin(['Pathogens','Cancer'])]

train_list = []
df_neo = df[df['Pathology']=='Neoantigen']
df_neo = df_neo[df_neo['Epitope.peptide']!='GILGFVFTL']
df_neo = df_neo.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first'}).reset_index()
df_neo['label'] = 'NeoAg'
train_list.append(df_neo)

df_maa = df[df['Pathology']=='Melanoma']
df_maa = df_maa.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first'}).reset_index()
df_maa['label'] = 'MAA'
train_list.append(df_maa)

df_path = df[df['Category']=='Pathogens']
path_sel = ['Influenza','Cytomegalovirus (CMV)','Epstein Barr virus (EBV)','Yellow fever virus']
df_path = df_path[df_path['Pathology'].isin(path_sel)]
df_path = df_path.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first','Pathology':'first'}).reset_index()
df_path['label'] = 'Viral'
# df_path['label'] = df_path['Pathology']
train_list.append(df_path)

df_train = pd.concat(train_list)
df_train = Process_Seq(df_train,'CDR3.beta.aa')

DTCR = DeepTCR_SS('Human_TIL')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

df_bms = pd.DataFrame()
df_bms['CDR3.beta.aa'] = DTCR.beta_sequences
df_bms['v_beta'] = DTCR.v_beta
df_bms['d_beta'] = DTCR.d_beta
df_bms['j_beta'] = DTCR.j_beta
df_bms['hla'] = DTCR.hla_data_seq
df_bms['freq'] = DTCR.freq
df_bms['sample'] = DTCR.sample_id
df_bms['class'] = DTCR.class_id


df_merge = pd.merge(df_train,df_bms,on='CDR3.beta.aa')

DTCR_inf = DeepTCR_WF('../human/HLA_TCR')
out = DTCR_inf.Sample_Inference(beta_sequences=np.array(df_merge['CDR3.beta.aa']),
                          v_beta=np.array(df_merge['v_beta']),
                          d_beta = np.array(df_merge['d_beta']),
                          j_beta = np.array(df_merge['j_beta']),
                          hla = np.array(df_merge['hla']))

df_merge['pred'] = out[:,0]
sns.violinplot(data=df_merge,x='label',y='pred',cut=0)
plt.ylabel('P(Response)',fontsize=24)
plt.tight_layout()
df_merge['w_pred'] = df_merge['pred']*df_merge['freq']

df_agg = df_merge.groupby(['sample','label']).agg({'freq':'sum','class':'first'}).reset_index()
df_agg = df_agg[df_agg['label']=='MAA']
sns.violinplot(data=df_agg,x='class',y='freq',cut=0)