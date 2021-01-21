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
cut_bottom = np.percentile(predicted[:,0],win)
cut_top = np.percentile(predicted[:,0],100-win)

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

df_post = pd.DataFrame()
df_post['beta'] = DTCR.beta_sequences
df_post['sample'] = DTCR.sample_id
df_post['gt'] = DTCR.class_id
df_post['freq'] = DTCR.freq
df_post['ID'] = df_post['sample'].map(post_dict)
df_post['seq_id'] = df_post['beta'] + '_' + df_post['ID'].astype(str)

df_merge = pd.merge(df_pre,df_post,on='seq_id')
df_merge['delta_freq'] = df_merge['freq_y'] - df_merge['freq_x']
df_merge = df_merge[df_merge['gt_x'] == 'crpr']
# x,y,z,_,_ = GKDE(np.array(df_merge['pred']),np.array(df_merge['delta_freq']))
# plt.scatter(x,y,c=z,cmap='jet')
df_top = df_merge[df_merge['pred'] > cut_top]
plt.hist(np.log10(df_top['delta_freq']),100)
df_bottom = df_merge[df_merge['pred'] < cut_bottom]
plt.hist(np.log10(df_bottom['delta_freq']),100)



df_merge['pred'][df_merge['pred']>cut_top] = 1.0
df_merge['pred'][df_merge['pred']<=cut_top] = 0.0

df_merge['w_pred_pre'] = df_merge['pred']*df_merge['freq_x']
df_merge['w_pred_post'] = df_merge['pred']*df_merge['freq_y']
df_agg = df_merge.groupby(['ID_x']).agg({'w_pred_pre':'sum','w_pred_post':'sum','gt_x':'first'}).reset_index()
df_agg['delta'] = df_agg['w_pred_post'] - df_agg['w_pred_pre']
#df_agg = pd.melt(df_agg,id_vars=['ID_x','gt_x'],value_vars=['w_pred_pre','w_pred_post'])
sns.violinplot(data=df_agg,x='gt_x',y='delta',cut=0)


