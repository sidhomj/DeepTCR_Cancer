import pandas as pd
import numpy as np
from DeepTCR.DeepTCR import DeepTCR_U
from DeepTCR.functions.data_processing import supertype_conv_op
import umap
import matplotlib.pyplot as plt
import scanpy as sc
def process_gene(gene,chain,df_merge):
    temp = df_merge['TR'+chain+gene+'_1'].str.split('TR'+chain+gene,expand=True)[1].str.split('-',expand=True)
    temp.fillna(value=1,inplace=True)
    temp[0] = temp[0].astype(int).map("{:02}".format)
    temp[1] = temp[1].astype(int).map("{:02}".format)
    temp[2] = 'TCR'+chain + gene + temp[0].astype(str) + '-'+temp[1].astype(str)
    return temp[2]

df_tcr = pd.read_csv('../../SC/scTCRs.csv')
df_tcr.drop(columns=df_tcr.columns[0],inplace=True)
df_tcr.rename(columns={df_tcr.columns[0]:'cell.barcode'},inplace=True)
df_tcr.dropna(subset=['CDR3B_1','CDR3A_1'],inplace=True)
df_tcr.rename(columns={df_tcr.columns[1]:'clonotype'},inplace=True)

df_label = pd.read_csv('../../SC/antigen.csv')
df_label.rename(columns={df_label.columns[0]:'clonotype'},inplace=True)
df_label['Cathegory'][df_label['Cathegory'] == 'Multi'] = 'NeoAg'
clonotype_dict = dict(zip(df_label['clonotype'],df_label['Cathegory']))
df_tcr['cat'] = df_tcr['clonotype'].map(clonotype_dict)
df_tcr['cat'].fillna('unknown',inplace=True)
df_tcr.reset_index(drop=True,inplace=True)

alpha_sequences = np.array(df_tcr['CDR3A_1'])
beta_sequences = np.array(df_tcr['CDR3B_1'])

DTCR = DeepTCR_U('unsup_sc')
DTCR.Load_Data(alpha_sequences=alpha_sequences,beta_sequences=beta_sequences)
DTCR.Train_VAE(latent_dim=64,accuracy_min=0.90,batch_size=1000,Load_Prev_Data=True)
features = DTCR.features
x2 = umap.UMAP().fit_transform(features)

df_tcr['x'] = x2[:,0]
df_tcr['y'] = x2[:,1]
df_tcr.sort_values(by='CDR3A_1',inplace=True)
df_tcr.drop_duplicates(subset=['CDR3A_1','CDR3B_1'],inplace=True)

color_dict = {}
color_dict['MAA'] = 'r'
color_dict['Viral'] = 'b'
color_dict['NeoAg'] = 'g'
color_dict['unknown'] = 'grey'

df = pd.DataFrame(np.array(df_tcr[['x','y']]))
df.index = np.array(df_tcr['cell.barcode'])
sc_obj = sc.AnnData(df)
sc_obj.obsm['X_umap'] = np.array(df_tcr[['x','y']])
sc_obj.obs['cat'] = np.array(df_tcr['cat'])
sc.pl.umap(sc_obj,color=['cat'],s=25,palette=color_dict)
plt.tight_layout()
sc_obj_antigen = sc_obj[sc_obj.obs['cat']!='unknown']
sc.pl.umap(sc_obj_antigen,color=['cat'],s=25,palette=color_dict)
plt.tight_layout()


