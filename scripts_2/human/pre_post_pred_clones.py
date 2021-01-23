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
def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r
def plot_pre_post(df_pre,df_post,label,cut_pre,cut_post,thresh=None):
    df_pre_sel = df_pre[df_pre['gt'] == label]
    df_post_sel = df_post[df_post['gt'] == label]
    if label == 'crpr':
        df_pre_sel = df_pre_sel[df_pre_sel['pred'] > cut_pre]
        df_post_sel = df_post_sel[df_post_sel['pred'] > cut_post]
    else:
        df_pre_sel = df_pre_sel[df_pre_sel['pred'] < cut_pre]
        df_post_sel = df_post_sel[df_post_sel['pred'] < cut_pre]

    df_sel_merge = pd.merge(df_pre_sel,df_post_sel,how='outer',on='seq_id')
    df_sel_merge['freq_x'].fillna(value=0.0,inplace=True)
    df_sel_merge['freq_y'].fillna(value=0.0,inplace=True)
    df_sel_merge = df_sel_merge[['seq_id','freq_x','freq_y']]
    df_sel_merge['delta'] = df_sel_merge['freq_y'] - df_sel_merge['freq_x']
    if thresh is not None:
        df_sel_merge = df_sel_merge[(df_sel_merge['freq_x'] > thresh) | (df_sel_merge['freq_y'] > thresh)]

    # your input data:
    befores = np.array(df_sel_merge['freq_x'])
    afters = np.array(df_sel_merge['freq_y'])

    # plotting the points
    fig,ax = plt.subplots()
    ax.scatter(np.zeros(len(befores)), befores)
    ax.scatter(np.ones(len(afters)), afters)

    # plotting the lines
    for i in range(len(befores)):
        ax.plot( [0,1], [befores[i], afters[i]], c='k')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Pre-Tx', 'Post-Tx'])

    return df_sel_merge

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

thresh = 0.001

#crpr
plot_pre_post(df_pre,df_post,label='crpr',cut_pre=cut_top_pre,cut_post=cut_top_post,thresh=thresh)
plt.title('CRPR')

#sdpd
plot_pre_post(df_pre,df_post,label='sdpd',cut_pre=cut_bottom_pre,cut_post=cut_bottom_post,thresh=thresh)
plt.title('SDPD')


