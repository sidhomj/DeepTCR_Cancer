"""Figure 3A,B"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve

model = 'TCR'
# model = 'HLA'
# model = 'TCR+HLA'
if model == 'TCR':
    file_base = 'sample_tcr'
elif model =='HLA':
    file_base = 'sample_hla'
elif model == 'TCR+HLA':
    file_base = 'sample_tcr_hla'

df_master = pd.read_csv('../../Data/other/Master_Beta.csv')
df_pre = pd.read_csv(file_base+'.csv')
df_master.dropna(subset=['Pre_Sample'],inplace=True)
pre_dict = dict(zip(df_master['Pre_Sample'],df_master['ID']))
df_pre['sample'] = df_pre['Samples'].map(pre_dict)

df_post = pd.read_csv(file_base+'_inf.csv')
df_master = pd.read_csv('../../Data/other/Master_Beta.csv')
df_master.dropna(subset=['Post_Sample'],inplace=True)
post_dict = dict(zip(df_master['Post_Sample'],df_master['ID']))
df_post['sample'] = df_post['Samples'].map(post_dict)

intersect = np.intersect1d(df_pre['sample'],df_post['sample'])
df_pre = df_pre[df_pre['sample'].isin(intersect)]
df_post = df_post[df_post['sample'].isin(intersect)]

#bootsrap AUC's
n_boots=5000
auc_list = []
model_list = []
for n in range(n_boots):
    idx = np.random.choice(range(len(df_pre)), len(df_pre), replace=True)
    auc_list.append(roc_auc_score(df_pre['y_test'].iloc[idx],df_pre['y_pred'].iloc[idx]))
    model_list.append('pre')

    idx = np.random.choice(range(len(df_post)), len(df_post), replace=True)
    auc_list.append(roc_auc_score(df_post['y_test'].iloc[idx],df_post['y_pred'].iloc[idx]))
    model_list.append('post')

df_bootstrap = pd.DataFrame()
df_bootstrap['model'] = model_list
df_bootstrap['auc'] = auc_list

#Draw figure
f, ax1 = plt.subplots()
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate',fontsize=18)
ax1.set_ylabel('True Positive Rate',fontsize=18)
roc_score = roc_auc_score(df_pre['y_test'], df_pre['y_pred'])
fpr, tpr, th = roc_curve(df_pre['y_test'], df_pre['y_pred'])
ax1.plot(fpr, tpr, lw=2, label='%s (%0.2f)' % ('Pre_Tx', roc_score))

roc_score = roc_auc_score(df_post['y_test'], df_post['y_pred'])
fpr, tpr, th = roc_curve(df_post['y_test'], df_post['y_pred'])
ax1.plot(fpr, tpr, lw=2, label='%s (%0.2f)' % ('Post_Tx', roc_score))
ax1.legend(loc='upper left',frameon=False)

ax2 = f.add_axes([0.48, .2, .4, .4])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
sns.violinplot(data=df_bootstrap,x='model',y='auc',ax=ax2,cut=0)
ax2.set_xlabel('')
plt.savefig(model+'pre_post_roc.eps')

#plot pre vs post preds
df_pre = df_pre.groupby(['Samples']).agg({'y_test':'first','y_pred':'mean','sample':'first'}).reset_index()
df_post = df_post.groupby(['Samples']).agg({'y_test':'first','y_pred':'mean','sample':'first'}).reset_index()
df_merge = pd.merge(df_pre,df_post,on='sample')
df_merge['response'] = None
df_merge['response'][df_merge['y_test_x']==1.0] = 'crpr'
df_merge['response'][df_merge['y_test_x']==0.0] = 'sdpd'

sns.scatterplot(data=df_merge,x='y_pred_x',y='y_pred_y',hue='response',s=100)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Pre-Tx Preds',fontsize=18)
plt.ylabel('Post-Tx Preds',fontsize=18)
plt.legend(loc='lower right',frameon=False,prop={'size':24})
plt.savefig(model+'_pre_post_preds.eps')

