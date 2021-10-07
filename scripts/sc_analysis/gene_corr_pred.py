import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from my_gsea.my_gsea import GSEA
import os
import seaborn as sns
from scipy.io import mmread

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores['cell.barcode'] = df_scores['patient']+'_'+ df_scores['cell.barcode'].str.split('_',expand=True)[1]

data = mmread('../../SC/data_all.mtx').T
barcodes = pd.read_csv('../../SC/barcodes.csv')
barcodes = np.array(barcodes.iloc[:,1])
genes = pd.read_csv('../../SC/genes.csv')
genes = np.array(genes.iloc[:,1])
data = pd.DataFrame(data.todense())
data.set_index(barcodes,inplace=True)
data.columns = genes

df_scores.set_index('cell.barcode',inplace=True)
df_merge2 = pd.merge(data,df_scores,right_index=True,left_index=True)

corr = []
for ii,g in enumerate(data.columns,0):
    c,p = spearmanr(df_merge2[g],df_merge2['pred'])
    corr.append(c)

df_corr = pd.DataFrame()
df_corr['gene'] = data.columns
df_corr['corr']= corr
# df_corr['corr'] = -df_corr['corr']
df_corr.sort_values(by='corr',ascending=False,inplace=True)
df_corr.dropna(inplace=True)

gs = GSEA()
# gs.Load_Gene_Sets(['c2.cp.biocarta.v7.4.symbols.gmt'])
gs_list = os.listdir('../../my_gsea/gene_sets')
gs.Load_Gene_Sets([gs_list[7]])
gs.Run_parallel(np.array(df_corr['gene']),np.array(df_corr['corr']),num_workers=8)
gs.Vis(gs.enr_results['Term'].iloc[0])
