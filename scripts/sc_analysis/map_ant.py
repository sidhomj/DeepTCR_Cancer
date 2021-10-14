import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df_antigen = pd.read_csv('../../SC/antigen.csv')
df_antigen['Cathegory'][df_antigen['Cathegory'] == 'Multi'] = 'NeoAg'

# df_antigen = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
# df_antigen = pd.melt(df_antigen)
# df_antigen.dropna(inplace=True)
# df_antigen.rename(columns={'value':'TCR clonotype family','variable':'Cathegory'},inplace=True)


df_merge = pd.merge(df_scores,df_antigen,on='TCR clonotype family')
df_merge.drop_duplicates(subset=['TCR clonotype family','CDR3B_1'],inplace=True)

sns.violinplot(data=df_merge,x='Cathegory',y='pred',
            order = list(df_merge.groupby(['Cathegory']).mean()['pred'].sort_values().index),
               cut=0)
sns.boxplot(data=df_merge,x='Cathegory',y='pred',
            order = list(df_merge.groupby(['Cathegory']).mean()['pred'].sort_values().index))

df_sel = df_merge[df_merge['Cathegory'] == 'Viral']
sns.violinplot(data=df_sel,x='NeoAg',y='pred',cut=0)

