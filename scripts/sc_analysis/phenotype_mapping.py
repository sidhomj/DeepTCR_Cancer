import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from my_gsea.my_gsea import GSEA
import os
import seaborn as sns

df = pd.read_csv('../../SC/umap.csv')
df_scores = pd.read_csv('tcrs_scored.csv')
df_scores['cell.barcode'] = df_scores['patient']+'_'+ df_scores['cell.barcode'].str.split('_',expand=True)[1]
df.rename(columns={df.columns[0]:'cell.barcode'},inplace=True)
df_merge = pd.merge(df,df_scores,on='cell.barcode')

plt.scatter(df_merge['UMAP_1'],df_merge['UMAP_2'],c=df_merge['pred'],cmap='jet',s=5)
sns.boxplot(data=df_merge,x='clusters',y='pred',order=list(df_merge.groupby(['clusters']).mean()['pred'].sort_values().index))

meta = pd.read_csv('../../SC/meta.csv')
data = pd.read_csv('../../SC/data.csv')
data.set_index('Unnamed: 0',inplace=True)
data = data.T

df_scores.set_index('cell.barcode',inplace=True)
df_merge2 = pd.merge(data,df_scores,right_index=True,left_index=True)

corr = []
for ii,g in enumerate(data.columns,0):
    c,p = spearmanr(df_merge2[g],df_merge2['pred'])
    corr.append(c)

df_corr = pd.DataFrame()
df_corr['gene'] = data.columns
df_corr['corr']= corr
df_corr.sort_values(by='corr',ascending=False,inplace=True)
df_corr.dropna(inplace=True)

gs = GSEA()
# gs.Load_Gene_Sets(['c2.cp.biocarta.v7.4.symbols.gmt'])
gs_list = os.listdir('../../my_gsea/gene_sets')
gs.Load_Gene_Sets([gs_list[12]])
gs.Run_parallel(np.array(df_corr['gene']),np.array(df_corr['corr']),num_workers=8)
gs.Vis(gs.enr_results['Term'].iloc[0])

