import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve

model = 'TCR'
model = 'HLA'
model = 'TCR+HLA'
if model == 'TCR':
    file_base = 'sample_tcr'
elif model =='HLA':
    file_base = 'sample_hla'
elif model == 'TCR+HLA':
    file_base = 'sample_tcr_hla'

df_master = pd.read_csv('Master_Beta.csv')
df_pre = pd.read_csv(file_base+'.csv')
df_master.dropna(subset=['Pre_Sample'],inplace=True)
pre_dict = dict(zip(df_master['Pre_Sample'],df_master['ID']))
df_pre['sample'] = df_pre['Samples'].map(pre_dict)
df_pre = df_pre.groupby(['Samples']).agg({'y_test':'first','y_pred':'mean','sample':'first'}).reset_index()

df_post = pd.read_csv(file_base+'_inf.csv')
df_master = pd.read_csv('Master_Beta.csv')
df_master.dropna(subset=['Post_Sample'],inplace=True)
post_dict = dict(zip(df_master['Post_Sample'],df_master['ID']))
df_post['sample'] = df_post['Samples'].map(pre_dict)
df_post = df_post.groupby(['Samples']).agg({'y_test':'first','y_pred':'mean','sample':'first'}).reset_index()

df_merge = pd.merge(df_pre,df_post,on='sample')
df_merge['response'] = None
df_merge['response'][df_merge['y_test_x']==1.0] = 'crpr'
df_merge['response'][df_merge['y_test_x']==0.0] = 'sdpd'

sns.scatterplot(data=df_merge,x='y_pred_x',y='y_pred_y',hue='response')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Pre-Tx Preds',fontsize=18)
plt.ylabel('Post-Tx Preds',fontsize=18)
plt.legend(loc='lower right',frameon=False)


f, ax1 = plt.subplots()
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate',fontsize=18)
ax1.set_ylabel('True Positive Rate',fontsize=18)
roc_score = roc_auc_score(df_merge['y_test_x'],df_merge['y_pred_x'])
fpr,tpr,th = roc_curve(df_merge['y_test_x'],df_merge['y_pred_x'])
class_name = 'Pre-Tx'
ax1.plot(fpr, tpr, lw=2, label='%s (%0.2f)' % (class_name, roc_score))

roc_score = roc_auc_score(df_merge['y_test_x'],df_merge['y_pred_y'])
fpr,tpr,th = roc_curve(df_merge['y_test_x'],df_merge['y_pred_y'])
class_name = 'Post-Tx'
ax1.plot(fpr, tpr, lw=2, label='%s (%0.2f)' % (class_name, roc_score))
ax1.legend(loc='lower right',frameon=False)




