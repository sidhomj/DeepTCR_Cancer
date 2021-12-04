from DeepTCR.DeepTCR import DeepTCR_WF
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score,roc_curve
import glob
import pandas as pd
from multiprocessing import Pool

gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

DTCR = DeepTCR_WF('post')
DTCR.Get_Data(directory='../../Data/bulk_tcr/post',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/bulk_tcr/post/HLA_Ref_sup_AB.csv')
beta_sequences = DTCR.beta_sequences
v_beta = DTCR.v_beta
d_beta = DTCR.d_beta
j_beta = DTCR.j_beta
hla = DTCR.hla_data_seq
sample_id = DTCR.sample_id
counts = DTCR.counts

df_master = pd.read_csv('../../Data/other/Master_Beta.csv')
df_master.dropna(subset=['Pre_Sample'],inplace=True)
sample_dict = dict(zip(df_master['Pre_Sample'],df_master['ID']))
df_master = pd.read_csv('../../Data/other/Master_Beta.csv')
df_master.dropna(subset=['Post_Sample'],inplace=True)
sample_dict.update(dict(zip(df_master['Post_Sample'],df_master['ID'])))
id_to_sample_dict = dict(zip(df_master['ID'],df_master['Post_Sample']))

pre_preds = pd.read_csv('sample_tcr.csv')
pre_preds['sample'] = pre_preds['Samples'].map(sample_dict)
pre_preds['cat'] = None
pre_preds['cat'][pre_preds['y_test']==1] = 'crpr'
pre_preds['cat'][pre_preds['y_test']==0] = 'sdpd'
pre_preds['model'] = None
for ii,_ in enumerate(range(0,600,6),0):
    pre_preds['model'].iloc[_:_+6] = 'model_'+str(ii)

pt_id = np.array(list(map(sample_dict.get,sample_id)))
pre_preds = pre_preds[pre_preds['sample'].isin(np.unique(pt_id))]

DTCR = DeepTCR_WF('TCR')
DFs = []
p = Pool(40)
for m in np.unique(pre_preds['model']):
    print(m)
    sel = pre_preds[pre_preds['model']==m]
    sel_idx = np.where(np.isin(pt_id,np.array(sel['sample'])))[0]
    DTCR.Sample_Inference(sample_labels=pt_id[sel_idx],
                          beta_sequences=beta_sequences[sel_idx],
                          v_beta=v_beta[sel_idx],
                          d_beta=d_beta[sel_idx],
                          j_beta=j_beta[sel_idx],
                          hla=hla[sel_idx],
                          counts=counts[sel_idx],
                          models=[m],
                          p=p)
    df_pred = DTCR.Inference_Pred_Dict['crpr']
    p_dict = dict(zip(df_pred['Samples'],df_pred['Pred']))
    sel['post_pred'] = sel['sample'].map(p_dict)
    sel['post_sample'] = sel['sample'].map(id_to_sample_dict)
    DFs.append(sel)

p.join()
p.close()
df_pred = pd.concat(DFs)
df_pred = df_pred[['post_sample','post_pred','y_test']]
df_pred = df_pred.rename(columns={'post_pred':'y_pred','post_sample':'Samples'})
df_pred.to_csv('sample_tcr_inf.csv',index=False)
