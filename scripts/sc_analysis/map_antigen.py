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
df_antigen['Cathegory'][df_antigen['Cathegory'] == 'Multi'] = 'NeoAg'

df_antigen = pd.melt(df_antigen,id_vars=[df_antigen.columns[0]],
               value_vars=['NeoAg', 'Cathegory','NeoAg','norm TCR activation', 'Avidity'])
#
# df_antigen = pd.pivot_table(df_antigen,index='norm TCR ac,columns='variable')
# df_antigen = pd.melt(df_antigen,id_vars=[df_antigen.columns[0],'Cathegory'],
#                value_vars=['NeoAg','NeoAg','norm TCR activation', 'Avidity'])

df_merge = pd.merge(df_spec,df_antigen,on='TCR clonotype family')

df_antigen_type = df_antigen[df_antigen['variable']=='Cathegory']
df_antigen_type = df_antigen_type[['TCR clonotype family','value']]
df_antigen_type.rename(columns={'value':'antigen_category'},inplace=True)
df_antigen_type.reset_index(drop=True,inplace=True)
df_antigen_type['antigen_category'][df_antigen_type['antigen_category'] == 'Multi'] = 'NeoAg'

df_merge = pd.merge(df_scores,df_antigen_type,on='TCR clonotype family')

df_merge = pd.merge(df_scores,df_antigen,on='TCR clonotype family')
# df_merge.drop_duplicates(subset=['TCR clonotype family'],inplace=True)

# sns.boxplot(data=df_merge,x='antigen_category',y='pred',
#             order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index))
sns.violinplot(data=df_merge,x='Cathegory',y='pred',
            order = list(df_merge.groupby(['Cathegory']).mean()['pred'].sort_values().index),
               cut=0)

sns.violinplot(data=df_merge,x='antigen_category',y='pred',
            order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index),
               cut=0)

df_sel = df_merge[df_merge['Cathegory'] == 'Viral']
df_sel.drop_duplicates(subset=['TCR clonotype family','CDR3B_1'],inplace=True)
sns.violinplot(data=df_sel,x='NeoAg',y='pred',
               cut=0)

df_antigen_type = df_spec
df_antigen_type = df_antigen_type[['TCR clonotype family','variable']]
df_antigen_type.rename(columns={'variable':'antigen_category'},inplace=True)
df_antigen_type.reset_index(drop=True,inplace=True)

df_merge = pd.merge(df_scores,df_antigen_type,on='TCR clonotype family')
sns.boxplot(data=df_merge,x='antigen_category',y='pred',
            order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index))
sns.violinplot(data=df_merge,x='antigen_category',y='pred',
            order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index),
            cut=0)

df_merge['CDR3B_1'].value_counts()

# lab = 'norm TCR activation'
# df_antigen_avidity = df_antigen[df_antigen['variable']==lab]
# df_antigen_avidity.dropna(inplace=True)
# df_antigen_avidity = df_antigen_avidity[['TCR clonotype family','value']]
# df_antigen_avidity.rename(columns={'value':lab},inplace=True)
# df_antigen_avidity.reset_index(drop=True,inplace=True)
# df_merge = pd.merge(df_scores,df_antigen_avidity,on='TCR clonotype family')
# # sns.boxplot(data=df_merge,x='antigen_category',y='pred',
# #             order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index))
# sns.violinplot(data=df_merge,x='antigen_category',y='pred',
#             order = list(df_merge.groupby(['antigen_category']).mean()['pred'].sort_values().index),
#                cut=0)
# sns.scatterplot(data=df_merge,x=lab,y='pred')
# from scipy.stats import spearmanr
# spearmanr(df_merge[lab],df_merge['pred'])
#
