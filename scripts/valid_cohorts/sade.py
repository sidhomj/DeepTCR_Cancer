from DeepTCR.DeepTCR import DeepTCR_WF
import pandas as pd
import numpy as np
from DeepTCR.functions.data_processing import supertype_conv_op
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

def process_gene(gene,chain,df_merge):
    temp = df_merge[gene.lower()+'_'+chain].str.split('TR'+chain+gene,expand=True)[1].str.split('-',expand=True)
    temp.fillna(value=1,inplace=True)
    temp[0] = temp[0].astype(int).map("{:02}".format)
    temp[1] = temp[1].astype(int).map("{:02}".format)
    temp[2] = 'TCR'+chain + gene + temp[0].astype(str) + '-'+temp[1].astype(str)
    return temp[2]

df_tcrs = pd.read_csv('../../Data/sade-feldman/sade-feldman_tcrs.csv')
df_tcrs = df_tcrs[['CDR3 (AA) - beta or delta chain',
                   'beta\delta V',
                   'beta\delta J',
                   'beta\delta D',
                   'HLA-A - allele 1', 'HLA-A - allele 2',
                   'HLA-B - allele 1', 'HLA-B - allele 2', 'HLA-C - allele 1',
                   'HLA-C - allele 2', 'sample_name',
                    'CD8 cluster - 6 clusters', 'CD8 cluster - 2 clusters'
                   ]]

df_tcrs.rename(columns={
    'CDR3 (AA) - beta or delta chain':'cdr3b',
    'beta\delta V':'v_B',
    'beta\delta J':'j_B',
    'beta\delta D':'d_B',
    'HLA-A - allele 1':'0',
    'HLA-A - allele 2':'1',
    'HLA-B - allele 1':'2',
    'HLA-B - allele 2':'3',
    'HLA-C - allele 1':'4',
    'HLA-C - allele 2':'5',
    'CD8 cluster - 6 clusters':'cluster_6',
    'CD8 cluster - 2 clusters':'cluster_2'
},inplace=True)

#remove delta TCRS
df_tcrs = df_tcrs[~df_tcrs['v_B'].str.startswith('TRD')]

df_tcrs['v_B'] = process_gene('V','B',df_tcrs)
df_tcrs['j_B'] = process_gene('J','B',df_tcrs)

#format hla
for _ in range(6):
    _ = str(_)
    df_tcrs[_] = df_tcrs[_].str[4:11].str.replace('_','').str.upper()

#get response data
df_response = pd.read_csv('../../Data/sade-feldman/response.csv')
df_response[df_response.columns[2]] = df_response[df_response.columns[2]].str.strip()
label_dict = dict(zip(df_response[df_response.columns[1]],df_response[df_response.columns[2]]))
df_tcrs['response'] = df_tcrs['sample_name'].map(label_dict)

#divide by pre/post
df_tcrs['time'] = df_tcrs['sample_name'].str.split('_',expand=True)[0]
df_tcrs.dropna(subset=['response'],inplace=True)
df_tcrs['counts'] = 1

beta_sequences = np.array(df_tcrs['cdr3b'])
v_beta = np.array(df_tcrs['v_B'])
d_beta = np.array(df_tcrs['d_B'])
j_beta = np.array(df_tcrs['j_B'])
hla = np.array(df_tcrs[['0','1','2','3','4','5']])
hla = np.array(supertype_conv_op(hla))
sample_labels= np.array(df_tcrs['sample_name'])
DTCR = DeepTCR_WF('../models/HLA_TCR')
out = DTCR.Sample_Inference(sample_labels=sample_labels,beta_sequences=beta_sequences,v_beta=v_beta,d_beta=d_beta,j_beta=j_beta,hla=hla)

df_preds = DTCR.Inference_Pred_Dict['crpr'][:]
df_preds['label'] = df_preds['Samples'].map(label_dict)
df_preds.sort_values(by='Pred',inplace=True,ascending=False)
df_preds['response_bin'] = LabelEncoder().fit_transform(df_preds['label'])
df_preds['time'] = df_preds['Samples'].str.split('_',expand=True)[0]
df_preds = df_preds[df_preds['time']=='Pre']
df_preds = df_preds[['Samples', 'Pred', 'label', 'response_bin']]
df_preds.to_csv('sade_preds.csv',index=False)
check=1