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
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.stats import spearmanr, pearsonr, fisher_exact
from statsmodels.stats.multitest import multipletests

df_sample = pd.read_csv('../../Data/other/Master_Beta.csv')
df_sample.dropna(subset=['Pre_Sample','Post_Sample'],inplace=True)
df_sample['ID'] = df_sample['ID'].astype(str)
pre_dict = dict(zip(df_sample['Pre_Sample'],df_sample['ID']))
post_dict = dict(zip(df_sample['Post_Sample'],df_sample['ID']))

DTCR = DeepTCR_WF('../models/HLA_TCR')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('../models/cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

df_pre = pd.DataFrame()
df_pre['beta'] = DTCR.beta_sequences
df_pre['sample'] = DTCR.sample_id
df_pre['pred'] = predicted[:,0]
df_pre['gt'] = DTCR.class_id
df_pre['freq'] = DTCR.freq
df_pre['counts'] = DTCR.counts
df_pre['ID'] = df_pre['sample'].map(pre_dict)
df_pre.dropna(subset=['ID'],inplace=True)
df_pre['seq_id'] = df_pre['beta'] + '_' + df_pre['ID'].astype(str)

DTCR = DeepTCR_WF('../models/post')
DTCR.Get_Data(directory='../../Data/bulk_tcr/post',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data_Post/HLA_Ref_sup_AB.csv')

with open('../models/cm038_ft_pred_inf.pkl','rb') as f:
    features,predicted = pickle.load(f)

df_post = pd.DataFrame()
df_post['beta'] = DTCR.beta_sequences
df_post['sample'] = DTCR.sample_id
df_post['pred'] = predicted[:,0]
df_post['gt'] = DTCR.class_id
df_post['freq'] = DTCR.freq
df_post['counts'] = DTCR.counts
df_post['ID'] = df_post['sample'].map(post_dict)
df_post.dropna(subset=['ID'],inplace=True)
df_post['seq_id'] = df_post['beta'] + '_' + df_post['ID'].astype(str)

df_merge = pd.merge(df_pre,df_post,how='outer',on='seq_id')
df_merge['sample'] = df_merge['seq_id'].str.split('_',expand=True)[1]
df_merge['freq_x'].fillna(value=0.0, inplace=True)
df_merge['freq_y'].fillna(value=0.0, inplace=True)
df_merge['counts_x'].fillna(value=0.0, inplace=True)
df_merge['counts_y'].fillna(value=0.0, inplace=True)
df_merge = df_merge[['sample','seq_id','gt_x','gt_y','pred_x','pred_y','freq_x', 'freq_y','counts_x','counts_y']]
df_merge['gt_y'].fillna(df_merge['gt_x'],inplace=True)
df_merge['gt_x'].fillna(df_merge['gt_y'],inplace=True)
df_merge['pred_y'].fillna(df_merge['pred_x'],inplace=True)
df_merge['pred_x'].fillna(df_merge['pred_y'],inplace=True)
df_merge = df_merge[['sample','seq_id','gt_x','pred_x','freq_x','freq_y','counts_x','counts_y']]
df_merge.rename(columns={'gt_x':'gt','pred_x':'pred',
                         'freq_x':'freq_pre',
                         'freq_y':'freq_post',
                         'counts_x':'counts_pre',
                         'counts_y':'counts_post'},inplace=True)
df_merge['delta'] = df_merge['freq_post'] - df_merge['freq_pre']
df_merge['abs_delta'] = np.abs(df_merge['delta'])
df_merge['fc'] = df_merge['freq_post']/df_merge['freq_pre']
df_merge['perc_change'] = (df_merge['freq_post']-df_merge['freq_pre'])/df_merge['freq_pre']
df_merge = pd.merge(df_pre,df_post,how='outer',on='seq_id')
df_merge['sample'] = df_merge['seq_id'].str.split('_',expand=True)[1]
df_merge['freq_x'].fillna(value=0.0, inplace=True)
df_merge['freq_y'].fillna(value=0.0, inplace=True)
df_merge['counts_x'].fillna(value=0.0, inplace=True)
df_merge['counts_y'].fillna(value=0.0, inplace=True)
df_merge = df_merge[['sample','seq_id','gt_x','gt_y','pred_x','pred_y','freq_x', 'freq_y','counts_x','counts_y']]
df_merge['gt_y'].fillna(df_merge['gt_x'],inplace=True)
df_merge['gt_x'].fillna(df_merge['gt_y'],inplace=True)
df_merge['pred_y'].fillna(df_merge['pred_x'],inplace=True)
df_merge['pred_x'].fillna(df_merge['pred_y'],inplace=True)
df_merge = df_merge[['sample','seq_id','gt_x','pred_x','freq_x','freq_y','counts_x','counts_y']]
df_merge.rename(columns={'gt_x':'gt','pred_x':'pred',
                         'freq_x':'freq_pre',
                         'freq_y':'freq_post',
                         'counts_x':'counts_pre',
                         'counts_y':'counts_post'},inplace=True)
df_merge['delta'] = df_merge['freq_post'] - df_merge['freq_pre']
df_merge['abs_delta'] = np.abs(df_merge['delta'])
df_merge['fc'] = df_merge['freq_post']/df_merge['freq_pre']
df_merge['perc_change'] = (df_merge['freq_post']-df_merge['freq_pre'])/df_merge['freq_pre']
df_merge['sample'].astype(int).astype(str)
df_sum = df_merge.groupby(['sample']).agg({'counts_pre':'sum','counts_post':'sum','gt':'first'})
df_sum['counts_pre'] = df_sum['counts_pre'].astype(int)
df_sum['counts_post'] = df_sum['counts_post'].astype(int)
counts_pre_dict = dict(zip(df_sum.index,df_sum['counts_pre']))
counts_post_dict = dict(zip(df_sum.index,df_sum['counts_post']))
df_merge['counts_pre_total'] = df_merge['sample'].map(counts_pre_dict)
df_merge['counts_post_total'] = df_merge['sample'].map(counts_post_dict)
# df_merge['OR'],df_merge['p_val'] = zip(*df_merge.apply(lambda x:
#                                    fisher_exact([[x['counts_post'],x['counts_pre']],
#                                                  [x['counts_post_total'],x['counts_pre_total']]]),
#                                    axis=1))
# _,df_merge['p_val_adj'],_,_ = multipletests(df_merge['p_val'],method='fdr_bh')
with open('df_dynamics.pkl','wb') as f:
    pickle.dump(df_merge,f,protocol=4)
