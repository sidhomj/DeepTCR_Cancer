import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
df = pd.melt(df)
df.dropna(inplace=True)
df.rename(columns={'value':'TCR clonotype family'},inplace=True)
df = df[df['variable'] != 'Tumor/control reactive']
df = df[['TCR clonotype family','variable']]
df.rename(columns={'variable':'tumor_spec'},inplace=True)
df1 = df

df_merge = pd.merge(df_scores,df,on='TCR clonotype family')
sns.violinplot(data=df_merge,x='tumor_spec',y='pred',
            order = list(df_merge.groupby(['tumor_spec']).mean()['pred'].sort_values().index),
            cut=0)

df = pd.read_csv('../../SC/antigen.csv')
df = pd.melt(df,id_vars=[df.columns[0]],
               value_vars=['NeoAg', 'Cathegory','norm TCR activation', 'Avidity'])


df = df[df['variable']=='Cathegory']
df = df[['TCR clonotype family','value']]
df.rename(columns={'value':'antigen_category'},inplace=True)
df.reset_index(drop=True,inplace=True)
df['antigen_category'][df['antigen_category'] == 'Multi'] = 'NeoAg'
df2 = df
df_merge = pd.merge(df_scores,df,on='TCR clonotype family')
sns.violinplot(data=df_merge,x='antigen_category',y='pred',
            order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index),
               cut=0)

######
df_merge = pd.merge(df1,df2,on='TCR clonotype family')
df_merge = pd.merge(df_merge,df_scores,on='TCR clonotype family')
sns.violinplot(data=df_merge,x='tumor_spec',y='pred',hue='antigen_category',cut=0)
