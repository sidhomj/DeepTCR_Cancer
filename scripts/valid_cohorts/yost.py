'''
This script runs the pre-trained DeepTCR repertoire model to predict response on the yost dataset and saves the predictions to a file to be used later by plot_auc.py to visualize performance.
'''
from DeepTCR.DeepTCR import DeepTCR_WF
import pandas as pd
import numpy as np
from DeepTCR.functions.data_processing import supertype_conv_op
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import percentileofscore

def process_gene(gene,chain,df_merge):
    temp = df_merge[gene.lower()+'_'+chain].str.split('TR'+chain+gene,expand=True)[1].str.split('-',expand=True)
    temp.fillna(value=1,inplace=True)
    temp[0] = temp[0].astype(int).map("{:02}".format)
    temp[1] = temp[1].astype(int).map("{:02}".format)
    temp[2] = 'TCR'+chain + gene + temp[0].astype(str) + '-'+temp[1].astype(str)
    return temp[2]

DTCR_load = DeepTCR_WF('load')
DTCR_load.Get_Data('../../Data/yost/data',Load_Prev_Data=False,
                   aa_column_beta=1, count_column=2, v_beta_column=7, d_beta_column=14, j_beta_column=21, data_cut=1.0)

df_all = pd.DataFrame()
df_all['files'] = DTCR_load.sample_list
ex = df_all['files'].str.split('_',expand=True)
df_all['Patient'] = ex[0]
df_all['Tumor Type'] = ex[1]
df_all['time'] = ex[2]
df_all['time'] = df_all['time'].str.replace('\d+', '')

df_meta = pd.read_csv('../../Data/yost/response.csv')
df_meta['Patient'] = df_meta['Patient'].str[0:5]
df_meta['Response'] = df_meta['Response'].str.split(' ',expand=True)[0]

df_merge = pd.merge(df_all,df_meta,on=['Patient','Tumor Type'],how='inner')
df_merge['sample_id'] =  df_merge['Patient']+'_'+df_merge['Tumor Type'] + '_' + df_merge['time']
df_merge = df_merge[df_merge['time']=='pre']

sample_dict = dict(zip(df_merge['files'],df_merge['sample_id']))
label_dict = dict(zip(df_merge['files'],df_merge['Response']))

idx = np.isin(DTCR_load.sample_id,df_merge['files'])

beta_sequences = DTCR_load.beta_sequences[idx]
v_beta = DTCR_load.v_beta[idx]
d_beta = DTCR_load.d_beta[idx]
j_beta = DTCR_load.j_beta[idx]
freq = DTCR_load.freq[idx]
sample_labels = np.array(list(map(sample_dict.get,DTCR_load.sample_id[idx])))
class_labels = np.array(list(map(label_dict.get,DTCR_load.sample_id[idx])))

DTCR = DeepTCR_WF('../models/TCR')
out = DTCR.Sample_Inference(sample_labels=sample_labels,beta_sequences=beta_sequences,v_beta=v_beta,d_beta=d_beta,j_beta=j_beta)

label_dict2 = dict(zip(df_merge['sample_id'],df_merge['Response']))
df_preds = DTCR.Inference_Pred_Dict['crpr'][:]
df_preds['label'] = df_preds['Samples'].map(label_dict2)
df_preds.sort_values(by='Pred',inplace=True,ascending=False)
df_preds['response_bin'] = LabelEncoder().fit_transform(df_preds['label'])
df_preds.to_csv('yost_preds.csv',index=False)

sns.boxplot(data=df_preds,x='label',y='Pred')

n_boots = 5000
scores = []
for _ in range(n_boots):
    try:
        df_temp = df_preds.sample(len(df_preds),replace=True)
        scores.append(roc_auc_score(df_temp['response_bin'], df_temp['Pred']))
    except:
        continue

fig,ax = plt.subplots(figsize=(5,5))
score = roc_auc_score(df_preds['response_bin'],df_preds['Pred'])
fpr,tpr,_ = roc_curve(df_preds['response_bin'],df_preds['Pred'])
key = 'crpr'
ax.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (key, score))
ax.set_xlabel('False Positive Rate',fontsize=16)
ax.set_ylabel('True Positive Rate',fontsize=16)
ax.set_xlim([0,1])
ax.set_ylim([0,1.05])
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc='lower right')

ax2 = fig.add_axes([0.62, .25, .25, .25])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
sns.violinplot(data=scores,ax=ax2,cut=0)
ax2.set_xlabel('')
ax2.hlines(y=0.5,xmin=ax2.get_xlim()[0],xmax=ax2.get_xlim()[1],linewidth=2,linestyles='--',color='red')

percentileofscore(scores,0.5)

df_preds.to_csv('yost_preds.csv',index=False)