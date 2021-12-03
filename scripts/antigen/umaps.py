import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

df_scores = pd.read_csv('oliveira_tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df = pd.read_csv('../../Data/sc/antigen.csv')
df = pd.melt(df,id_vars=[df.columns[0]],
               value_vars=['NeoAg', 'Cathegory','norm TCR activation', 'Avidity'])

df = df[df['variable']=='Cathegory']
df = df[['TCR clonotype family','value']]
df.rename(columns={'value':'label'},inplace=True)
df.reset_index(drop=True,inplace=True)
df['label'][df['label'] == 'Multi'] = 'NeoAg'
df2 = df
df_merge = pd.merge(df_scores,df,on='TCR clonotype family')
label_dict = {'MAA':'Tumor Specific', 'NeoAg':'Tumor Specific', 'Viral':'Virus Specific'}
df_merge['label'] = df_merge['label'].map(label_dict)
df_umap = pd.read_csv('../../Data/sc/umap.csv')
df_umap.rename(columns={'Unnamed: 0':'cell.barcode 2'},inplace=True)
df_merge = pd.merge(df_merge,df_umap,on='cell.barcode 2')
labels = np.unique(df_merge['label'])
num_windows = len(labels)+1
fig,ax = plt.subplots(1,num_windows,figsize=(num_windows*3,3))
for ii,g in enumerate(labels,0):
    ax[ii].scatter(df_umap['UMAP_1'],df_umap['UMAP_2'],c='grey',s=1)
    idx = df_merge['label'] == g
    ax[ii].scatter(df_merge['UMAP_1'][idx],df_merge['UMAP_2'][idx],c='r',s=1)
    ax[ii].set_title(g)
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])

idx = np.argsort(np.array(df_merge['pred']))
ii = ii+1
im = ax[ii].scatter(df_merge['UMAP_1'][idx],df_merge['UMAP_2'][idx],c=df_merge['pred'][idx],s=5,cmap='jet')
fig.colorbar(im)
ax[ii].set_title('P(Response)')
ax[ii].set_xticks([])
ax[ii].set_yticks([])
plt.tight_layout()
plt.savefig('umap.png',dpi=1200)

# plt.tight_layout()

# low_color = 'blue'
# high_color = 'red'
# cvals = [-1,1]
# colors = [low_color,high_color]
# norm = plt.Normalize(min(cvals), max(cvals))
# tuples = list(zip(map(norm, cvals), colors))
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)