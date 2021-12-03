import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import gaussian_kde
def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r

df_scores = pd.read_csv('oliveira_tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df_antigen = pd.read_csv('../../Data/sc/antigen.csv')
df_antigen['Cathegory'][df_antigen['Cathegory'] == 'Multi'] = 'NeoAg'

train_list = []

df_viral = df_antigen[df_antigen['Cathegory'] == 'Viral']
df_viral['label'] = df_viral['NeoAg']
train_list.append(df_viral)

df_maa = df_antigen[df_antigen['Cathegory'] == 'MAA']
df_maa = df_maa[~df_maa['NeoAg'].isin(['PMEL589-603','Pmel pool','TYR207-215',
                                       'PMEL154-162','TYR514-524','PMEL209-217'])]
df_maa['label'] = df_maa['NeoAg']
train_list.append(df_maa)

df_neo = df_antigen[df_antigen['Cathegory']=='NeoAg']
df_neo['label'] = df_neo['Cathegory']
train_list.append(df_neo)

df_antigen = pd.concat(train_list)

df_merge = pd.merge(df_scores,df_antigen,on='TCR clonotype family')
df_merge.drop_duplicates(subset=['TCR clonotype family','CDR3B_1'],inplace=True)

# pal = {'NeoAg':'b',''}
sns.violinplot(data=df_merge,x='label',y='pred',
            order = list(df_merge.groupby(['label']).mean()['pred'].sort_values().index),
               cut=0)
plt.ylabel('P(Response)',fontsize=24)
plt.xlabel('')