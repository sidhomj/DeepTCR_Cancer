from DeepTCR.DeepTCR import DeepTCR_WF
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score,roc_curve
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import mannwhitneyu
def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r

gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

df_sample = pd.read_csv('Master_Beta.csv')
pre_dict = dict(zip(df_sample['Pre_Sample'],df_sample['ID']))
post_dict = dict(zip(df_sample['Post_Sample'],df_sample['ID']))


DTCR = DeepTCR_WF('Human_TIL',device='/device:GPU:0')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

win = 10
cut_bottom_pre = np.percentile(predicted[:,0],win)
cut_top_pre = np.percentile(predicted[:,0],100-win)

df_pre = pd.DataFrame()
df_pre['beta'] = DTCR.beta_sequences
df_pre['sample'] = DTCR.sample_id
df_pre['pred'] = predicted[:,0]
df_pre['gt'] = DTCR.class_id
df_pre['freq'] = DTCR.freq
df_pre['ID'] = df_pre['sample'].map(pre_dict)
df_pre['seq_id'] = df_pre['beta'] + '_' + df_pre['ID'].astype(str)


DTCR = DeepTCR_WF('load')
DTCR.Get_Data(directory='../../Data_Post',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data_Post/HLA_Ref_sup_AB.csv')

with open('cm038_ft_pred_inf.pkl','rb') as f:
    features,predicted = pickle.load(f)

cut_bottom_post = np.percentile(predicted[:,0],win)
cut_top_post = np.percentile(predicted[:,0],100-win)

df_post = pd.DataFrame()
df_post['beta'] = DTCR.beta_sequences
df_post['sample'] = DTCR.sample_id
df_post['pred'] = predicted[:,0]
df_post['gt'] = DTCR.class_id
df_post['freq'] = DTCR.freq
df_post['ID'] = df_post['sample'].map(post_dict)
df_post['seq_id'] = df_post['beta'] + '_' + df_post['ID'].astype(str)

overlap_list = []
response_list = []
for s_ in np.unique(df_pre['ID']):
    lb = df_pre['gt'][df_pre['ID']==s_].iloc[0]
    response_list.append(lb)
    if lb == 'crpr':
        df_pre_sel = df_pre[df_pre['ID']==s_]
        df_pre_sel = df_pre_sel[df_pre_sel['pred'] > cut_top_pre]
        pre_freq = np.sum(df_pre_sel['freq'])

        df_post_sel = df_post[df_post['ID']==s_]
        df_post_sel = df_post_sel[df_post_sel['pred'] > cut_top_post]
        post_freq = np.sum(df_post_sel['freq'])

        df_merge_sel = pd.merge(df_pre_sel,df_post_sel,on='beta')
        post_freq_overlap = np.sum(df_merge_sel['freq_x'])/pre_freq
    else:
        df_pre_sel = df_pre[df_pre['ID']==s_]
        df_pre_sel = df_pre_sel[df_pre_sel['pred'] < cut_bottom_pre]
        pre_freq = np.sum(df_pre_sel['freq'])

        df_post_sel = df_post[df_post['ID']==s_]
        df_post_sel = df_post_sel[df_post_sel['pred'] < cut_bottom_post]
        post_freq = np.sum(df_post_sel['freq'])

        df_merge_sel = pd.merge(df_pre_sel,df_post_sel,on='beta')
        post_freq_overlap = np.sum(df_merge_sel['freq_x'])/pre_freq

    overlap_list.append(post_freq_overlap)

df_plot = pd.DataFrame()
df_plot['ID'] = np.unique(df_pre['ID'])
df_plot['sample'] = df_plot['ID'].map({v: k for k, v in pre_dict.items()})
df_plot['overlap'] = overlap_list
df_plot['label'] = response_list
s = pd.read_csv('sample_tcr_hla.csv')
s = s.groupby(['Samples']).agg({'y_pred':'mean','y_test':'mean'}).reset_index()
s.rename(columns={'y_pred': 'preds','Samples':'sample'}, inplace=True)
pred_dict = dict(zip(s['sample'],s['preds']))
df_plot['pred'] = df_plot['sample'].map(pred_dict)

ax = sns.violinplot(data=df_plot,x='label',y='overlap',cut=0)
x_ticks = ax.get_xticklabels()
ax.set_xticklabels(x_ticks, rotation=0, fontsize=18)
plt.xlabel('')
plt.ylabel('% Overlap',fontsize=18)
plt.ylim([0,1])
y_ticks = ax.get_yticklabels()
ax.set_yticklabels(y_ticks,fontsize=14)
plt.savefig('overlap.eps')

_,p_val = mannwhitneyu(df_plot['overlap'][df_plot['label']=='crpr'],df_plot['overlap'][df_plot['label']=='sdpd'])

plt.figure()
sns.scatterplot(data=df_plot[df_plot['label']=='crpr'],x='overlap',y='pred')
plt.title('crpr')

plt.figure()
sns.scatterplot(data=df_plot[df_plot['label']=='sdpd'],x='overlap',y='pred')
plt.title('sdpd')

