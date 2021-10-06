import pandas as pd
import numpy as np
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_SS

df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)

df_spec = pd.read_csv('../../SC/TCR clones tumor specificity.csv')
df_spec = pd.melt(df_spec)
df_spec.dropna(inplace=True)
df_spec.rename(columns={'value':'TCR clonotype family'},inplace=True)

df_antigen = pd.read_csv('../../SC/antigen.csv')
df_antigen = pd.melt(df_antigen,id_vars=[df_antigen.columns[0]],
               value_vars=['NeoAg', 'Cathegory','norm TCR activation', 'Avidity'])

# df_merge = pd.merge(df_spec,df_antigen,on='TCR clonotype family')

df_antigen_type = df_antigen[df_antigen['variable']=='Cathegory']
df_antigen_type = df_antigen_type[['TCR clonotype family','value']]
df_antigen_type.rename(columns={'value':'antigen_category'},inplace=True)
df_antigen_type.reset_index(drop=True,inplace=True)
df_antigen_type['antigen_category'][df_antigen_type['antigen_category'] == 'Multi'] = 'NeoAg'

df_merge = pd.merge(df_scores,df_antigen_type,on='TCR clonotype family')
sns.boxplot(data=df_merge,x='antigen_category',y='pred',
            order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index))


df_antigen_type = df_spec
df_antigen_type = df_antigen_type[['TCR clonotype family','variable']]
df_antigen_type.rename(columns={'variable':'antigen_category'},inplace=True)
df_antigen_type.reset_index(drop=True,inplace=True)

df_merge = pd.merge(df_scores,df_antigen_type,on='TCR clonotype family')
sns.boxplot(data=df_merge,x='antigen_category',y='pred',
            order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index))

df_merge['CDR3B_1'].value_counts()
