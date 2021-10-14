from DeepTCR.DeepTCR import DeepTCR_SS
import pandas as pd
import numpy as np
from DeepTCR.functions.data_processing import supertype_conv_op
def process_gene(gene,chain,df_merge):
    temp = df_merge['TR'+chain+gene+'_1'].str.split('TR'+chain+gene,expand=True)[1].str.split('-',expand=True)
    temp.fillna(value=1,inplace=True)
    temp[0] = temp[0].astype(int).map("{:02}".format)
    temp[1] = temp[1].astype(int).map("{:02}".format)
    temp[2] = 'TCR'+chain + gene + temp[0].astype(str) + '-'+temp[1].astype(str)
    return temp[2]

df_tcr = pd.read_csv('../../SC/scTCRs.csv')
df_tcr.dropna(subset=['CDR3B_1'],inplace=True)
df_tcr['patient'] = df_tcr['patient'].str.split('.',expand=True)[0]

df_hla = pd.read_csv('../../SC/hla.csv')
df_merge = pd.merge(df_tcr,df_hla,on='patient')
df_merge['TRBV_1'] = process_gene('V','B',df_merge)
df_merge['TRBJ_1'] = process_gene('J','B',df_merge)
df_merge['TRBD_1'][df_merge['TRBD_1']=='None'] = 'nan'

beta_sequences = np.array(df_merge['CDR3B_1'])
v_beta = np.array(df_merge['TRBV_1'])
d_beta = np.array(df_merge['TRBD_1'])
j_beta = np.array(df_merge['TRBJ_1'])
hla = np.array(df_merge[['0','1','2','3','4','5']])
hla = np.array(supertype_conv_op(hla))
DTCR = DeepTCR_SS('mcpas')
out = DTCR.Sequence_Inference(beta_sequences=beta_sequences)
                            # models=['model_'+str(x) for x in range(10)])
for ii,c in enumerate(DTCR.lb.classes_):
    df_merge[c] = out[:,ii]
df_merge.to_csv('tcrs_scored_mcpas.csv',index=False)

df_scores = df_merge
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df_antigen = pd.read_csv('../../SC/antigen.csv')
df_antigen['Cathegory'][df_antigen['Cathegory'] == 'Multi'] = 'NeoAg'

# df_antigen = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
# df_antigen = pd.melt(df_antigen)
# df_antigen.dropna(inplace=True)
# df_antigen.rename(columns={'value':'TCR clonotype family','variable':'Cathegory'},inplace=True)


df_merge = pd.merge(df_scores,df_antigen,on='TCR clonotype family')
df_merge.drop_duplicates(subset=['TCR clonotype family','CDR3B_1'],inplace=True)
import seaborn as sns
sns.violinplot(data=df_merge,x='Cathegory',y='NeoAg_x',
            order = list(df_merge.groupby(['Cathegory']).mean()['NeoAg_x'].sort_values().index),
               cut=0)

df = pd.read_csv('../../SC/umap.csv')
df_scores['cell.barcode'] = df_scores['cell.barcode 2']
df.rename(columns={df.columns[0]:'cell.barcode'},inplace=True)
df_merge = pd.merge(df,df_scores,on='cell.barcode')

import matplotlib.pyplot as plt
sel = 'MAA'
idx = np.argsort(np.array(df_merge[sel]))
plt.scatter(df_merge['UMAP_1'][idx],df_merge['UMAP_2'][idx],c=df_merge[sel][idx],cmap='jet',s=5)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()

sns.boxplot(data=df_merge,x='clusters',y=sel,order=list(df_merge.groupby(['clusters']).mean()[sel].sort_values().index))
plt.ylabel('P(Response)',fontsize=24)
plt.xlabel('Clusters',fontsize=25)
plt.tight_layout()