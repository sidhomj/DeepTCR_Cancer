import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS
import matplotlib.pyplot as plt

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
df = pd.melt(df)
df.dropna(inplace=True)
df.rename(columns={'value':'TCR clonotype family'},inplace=True)
df = df[df['variable'] != 'Tumor/control reactive']
df = df[['TCR clonotype family','variable']]
df.rename(columns={'variable':'label'},inplace=True)
df1 = df

df_merge = pd.merge(df_scores,df,on='TCR clonotype family')
sns.violinplot(data=df_merge,x='label',y='pred',
            order = list(df_merge.groupby(['label']).mean()['pred'].sort_values().index),
            cut=0)
plt.ylabel('P(Response)',fontsize=24)
plt.xticks(fontsize=16)
plt.xlabel(None)

df = pd.read_csv('../../SC/antigen.csv')
df = pd.melt(df,id_vars=[df.columns[0]],
               value_vars=['NeoAg', 'Cathegory','norm TCR activation', 'Avidity'])


df = df[df['variable']=='Cathegory']
df = df[['TCR clonotype family','value']]
df.rename(columns={'value':'label'},inplace=True)
df.reset_index(drop=True,inplace=True)
df['label'][df['label'] == 'Multi'] = 'NeoAg'
df2 = df
df_merge = pd.merge(df_scores,df,on='TCR clonotype family')
sns.violinplot(data=df_merge,x='label',y='pred',
            order = list(df_merge.groupby(['label']).mean()['pred'].sort_values().index),
               cut=0)
plt.ylabel('P(Response)',fontsize=24)
plt.xticks(fontsize=16)
plt.xlabel(None)

df_umap = pd.read_csv('../../SC/umap.csv')
df_umap.rename(columns={'Unnamed: 0':'cell.barcode 2'},inplace=True)
df_merge = pd.merge(df_merge,df_umap,on='cell.barcode 2')
labels = np.unique(df_merge['label'])
fig,ax = plt.subplots(1,len(labels),figsize=(len(labels)*3,3))
for ii,g in enumerate(labels,0):
    ax[ii].scatter(df_umap['UMAP_1'],df_umap['UMAP_2'],c='grey',s=1)
    idx = df_merge['label'] == g
    ax[ii].scatter(df_merge['UMAP_1'][idx],df_merge['UMAP_2'][idx],c='r',s=1)
    ax[ii].set_title(g)
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])
plt.tight_layout()

idx = np.argsort(np.array(df_merge['pred']))
plt.scatter(df_merge['UMAP_1'][idx],df_merge['UMAP_2'][idx],c=df_merge['pred'][idx],s=5,cmap='jet')
plt.xticks([])
plt.yticks(([]))
plt.colorbar()
plt.tight_layout()