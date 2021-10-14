import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DeepTCR.functions.data_processing import Process_Seq


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
df_train.rename(columns={'CDR3.beta.aa':'CDR3B_1'},inplace=True)

df_merge = pd.merge(df_merge,df_train,on='CDR3B_1')
fig,ax = plt.subplots(1,3,figsize=(10,4))
for ii,g in enumerate(np.unique(df_merge['label'])):
    df_sel = df_merge[df_merge['label']==g]
    ax[ii].scatter(df_all['UMAP_1'],df_all['UMAP_2'],s=5,c='grey')
    ax[ii].scatter(df_sel['UMAP_1'],df_sel['UMAP_2'],s=5,c='red')
    ax[ii].set_title(g)
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])
plt.tight_layout()
