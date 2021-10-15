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


df_scores = pd.read_csv('tcrs_scored.csv')
df_scores.rename(columns={'final.clonotype.family':'TCR clonotype family'},inplace=True)


df_antigen = pd.read_csv('../../SC/antigen.csv')
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

sns.violinplot(data=df_merge,x='label',y='pred',
            order = list(df_merge.groupby(['label']).mean()['pred'].sort_values().index),
               cut=0)
plt.ylabel('P(Response)',fontsize=24)

DTCR = DeepTCR_SS('HLA_TCR')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('../human/cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

file = 'cm038_x2_u.pkl'
featurize = False
with open(os.path.join('../human',file),'rb') as f:
    X_2 = pickle.load(f)

df_bms = pd.DataFrame()
df_bms['CDR3.beta.aa'] = DTCR.beta_sequences
df_bms['v_beta'] = DTCR.v_beta
df_bms['d_beta'] = DTCR.d_beta
df_bms['j_beta'] = DTCR.j_beta
df_bms['hla'] = DTCR.hla_data_seq
df_bms['freq'] = DTCR.freq
df_bms['sample'] = DTCR.sample_id
df_bms['class'] = DTCR.class_id
df_bms['x'] = X_2[:,0]
df_bms['y'] = X_2[:,1]
df_bms['pred'] = predicted[:,0]

df_merge.rename(columns={'CDR3B_1':'CDR3.beta.aa'},inplace=True)
df_merge = pd.merge(df_merge,df_bms,on='CDR3.beta.aa')

win = 10
cut_bottom = np.percentile(predicted[:,0],win)
cut_top = np.percentile(predicted[:,0],100-win)
df_merge = df_merge[(df_merge['pred'] > cut_top) | (df_merge['pred']< cut_bottom)]

plt.scatter(df_merge['x'],df_merge['y'])
xlim = plt.xlim()
ylim = plt.ylim()
plt.close()

order = ['Viral','Melan-A/MART-1']
order = np.unique(df_merge['label'])
fig,ax = plt.subplots(1,len(order),figsize=(len(order)*4,4))
for ii,l in enumerate(order):
    df_plot = df_merge[df_merge['label']==l]
    x,y,c,_,_ = GKDE(np.array(df_plot['x']),np.array(df_plot['y']))
    # x,y,c = np.array(df_plot['x']),np.array(df_plot['y']),df_plot['pred']
    ax[ii].scatter(x,y,c=c,cmap='jet',s=25)
    ax[ii].set_title(l)
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])
    ax[ii].set_xlim(xlim)
    ax[ii].set_ylim(ylim)
plt.tight_layout()