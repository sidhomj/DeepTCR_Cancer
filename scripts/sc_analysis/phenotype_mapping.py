import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from my_gsea.my_gsea import GSEA
import os
import seaborn as sns
from scipy.io import mmread

df = pd.read_csv('../../SC/umap.csv')
df_scores = pd.read_csv('tcrs_scored.csv')
df_scores['cell.barcode'] = df_scores['patient']+'_'+ df_scores['cell.barcode'].str.split('_',expand=True)[1]
df.rename(columns={df.columns[0]:'cell.barcode'},inplace=True)
df_merge = pd.merge(df,df_scores,on='cell.barcode')

idx = np.argsort(np.array(df_merge['pred']))
plt.scatter(df_merge['UMAP_1'][idx],df_merge['UMAP_2'][idx],c=df_merge['pred'][idx],cmap='jet',s=5)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.tight_layout()

sns.boxplot(data=df_merge,x='clusters',y='pred',order=list(df_merge.groupby(['clusters']).mean()['pred'].sort_values().index))
plt.ylabel('P(Response)',fontsize=24)
plt.xlabel('Clusters',fontsize=25)
plt.tight_layout()