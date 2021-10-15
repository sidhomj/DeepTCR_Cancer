"""Figure 3G"""
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
def plot_pre_post(df_pre,df_post,label,pred_label,cut_pre,cut_post,thresh=None,plot=True):
    df_pre_sel = df_pre[df_pre['gt'] == label]
    df_post_sel = df_post[df_post['gt'] == label]
    if pred_label == 'crpr':
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
    df_sel_merge['perc_change_pre'] = (df_sel_merge['freq_y'] - df_sel_merge['freq_x'])/df_sel_merge['freq_x']
    df_sel_merge['perc_change_post'] = (df_sel_merge['freq_x'] - df_sel_merge['freq_y'])/df_sel_merge['freq_y']
    # df_sel_merge['fc'] = df_sel_merge['freq_y']/df_sel_merge['freq_x']
    if thresh is not None:
        df_sel_merge = df_sel_merge[(df_sel_merge['freq_x'] > thresh) | (df_sel_merge['freq_y'] > thresh)]
        # df_sel_merge = df_sel_merge[df_sel_merge['freq_x'] > thresh]

    if plot:
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
        ax.set_xticklabels(['Pre-Tx', 'Post-Tx'],fontsize=24)
    else:
        ax = None

    return df_sel_merge,ax

gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

df_sample = pd.read_csv('Master_Beta.csv')
pre_dict = dict(zip(df_sample['Pre_Sample'],df_sample['ID']))
post_dict = dict(zip(df_sample['Post_Sample'],df_sample['ID']))


DTCR = DeepTCR_WF('HLA_TCR')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=True,
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
DTCR.Get_Data(directory='../../Data_Post',Load_Prev_Data=True,
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

thresh = 0.01

#crpr
df_crpr_r,ax = plot_pre_post(df_pre,df_post,label='crpr',pred_label='crpr',
                             cut_pre=cut_top_pre,cut_post=cut_top_post,thresh=thresh,
                             plot=False)
df_crpr_r['patient'] = df_crpr_r['seq_id'].str.split('_',expand=True)[1]
y_ticks = ax.get_yticklabels()
ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.title('CRPR',fontsize=24)

df_crpr_nr,ax = plot_pre_post(df_pre,df_post,label='crpr',pred_label='sdpd',
                              cut_pre=cut_bottom_pre,cut_post=cut_bottom_post,thresh=thresh,
                              plot=False)
df_crpr_nr['patient'] = df_crpr_nr['seq_id'].str.split('_',expand=True)[1]
y_ticks = ax.get_yticklabels()
ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.title('CRPR',fontsize=24)

# df_crpr_r['label'] = 'r'
# df_crpr_nr['label'] = 'nr'
# df_plot = pd.concat([df_crpr_r,df_crpr_nr])
# df_plot['delta'] = np.log10(df_plot['delta'])
# sns.violinplot(data=df_plot,x='label',y='delta',cut=0)
# df_plot.dropna(subset=['fc'],inplace=True)
# df_plot = df_plot[df_plot['freq_x']>0]
# df_plot['fc'] = np.log10(df_plot['fc'])
# sns.violinplot(data=df_plot,x='label',y='fc',cut=0)

plt.savefig('pre_post_clones_crpr.png',dpi=1200)

#sdpd
df_sdpd_r,ax = plot_pre_post(df_pre,df_post,label='sdpd',pred_label='crpr',
                             cut_pre=cut_top_pre,cut_post=cut_top_post,thresh=thresh,
                             plot=False)
df_sdpd_r['patient'] = df_sdpd_r['seq_id'].str.split('_',expand=True)[1]
y_ticks = ax.get_yticklabels()
ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.title('SDPD',fontsize=24)

df_sdpd_nr,ax = plot_pre_post(df_pre,df_post,label='sdpd',pred_label='sdpd',
                              cut_pre=cut_bottom_pre,cut_post=cut_bottom_post,thresh=thresh,
                              plot=False)
df_sdpd_nr['patient'] = df_sdpd_nr['seq_id'].str.split('_',expand=True)[1]
y_ticks = ax.get_yticklabels()
ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.title('SDPD',fontsize=24)


df_crpr_r['label'] = 'crpr'
df_sdpd_r['label'] = 'sdpd'
df_r = pd.concat([df_crpr_r,df_sdpd_r])
# df_r = df_r[df_r['freq_x']>0.0]
# df_r = df_r[df_r['freq_x']>0]
# df_r['delta'] = np.log10(df_r['delta'])
# df_r['perc_change_pre'] = np.log10(df_r['perc_change_pre'])
# df_agg = df_r.groupby(['patient']).agg({'perc_change_pre':'mean','label':'first'})
# sns.violinplot(data=df_r,x='label',y='delta',cut=0)
sns.boxplot(data=df_r,x='label',y='delta')


df_r = df_r[df_r['freq_y']>0.0]
# df_r = df_r[df_r['freq_x']>0]
# df_r['delta'] = np.log10(df_r['delta'])
df_r['perc_change_post'] = np.log10(df_r['perc_change_post'])
sns.violinplot(data=df_r,x='label',y='perc_change_post',cut=0)

df_crpr_nr['label'] = 'crpr'
df_sdpd_nr['label'] = 'sdpd'
df_nr = pd.concat([df_crpr_nr,df_sdpd_nr])
# df_agg = df_nr.groupby(['patient']).agg({'freq_x':'sum','freq_y':'sum','label':'first'})
# df_agg.sort_values(by='label',inplace=True)
# df_agg['delta'] = df_agg['freq_y'] - df_agg['freq_x']
# df_nr = df_nr[df_nr['freq_x']>0]
# df_nr['fc'] = np.log10(df_nr['fc'])
# df_nr['delta'] = np.log10(df_nr['delta'])
# sns.violinplot(data=df_nr,x='label',y='delta',cut=0)
sns.boxplot(data=df_nr,x='label',y='delta')

sns.violinplot(data=df_agg,x='label',y='delta',cut=0)
sns.violinplot(data=df_nr,x='label',y='fc',cut=0)



df_crpr_r['label'] = 'crpr'
df_sdpd_r['label'] = 'sdpd'
df_r = pd.concat([df_crpr_r,df_sdpd_r])
df_agg = df_r.groupby(['patient']).agg({'freq_x':'sum','freq_y':'sum','label':'first'})
df_agg.sort_values(by='label',inplace=True)
df_agg['delta'] = df_agg['freq_y'] - df_agg['freq_x']
# df_nr = df_nr[df_nr['freq_x']>0]
# df_nr['fc'] = np.log10(df_nr['fc'])
# df_nr['delta'] = np.log10(df_nr['delta'])
sns.violinplot(data=df_r,x='label',y='delta',cut=0)
sns.violinplot(data=df_agg,x='label',y='delta',cut=0)

df_crpr_r['label'] = 'crpr'
df_sdpd_r['label'] = 'sdpd'
df_r = pd.concat([df_crpr_r,df_sdpd_r])
# df_r['increase'] = df_r['delta'] > 0
df_r['increase'] = df_r['fc'] > 10000
df_r['count'] = 1
df_agg = df_r.groupby(['patient']).agg({'increase':'sum','count':'sum','label':'first'})
df_agg['percent_inc'] = df_agg['increase']/df_agg['count']
sns.violinplot(data=df_agg,x='label',y='percent_inc',cut=0)
df_r_pre = df_r[df_r['freq_x'] > 0.0]

sns.violinplot(data=df_r_pre,x='label',y='fc')
df_agg = df_r_pre.groupby(['patient']).agg({'fc':'mean','label':'first'})
sns.violinplot(data=df_agg,x='label',y='fc',cut=0)

df_agg = df_r.groupby(['patient']).agg({'freq_x':'sum','freq_y':'sum','label':'first'})
df_agg['delta'] = df_agg['freq_y'] /df_agg['freq_x']
sns.violinplot(data=df_agg,x='label',y='delta',cut=0)



