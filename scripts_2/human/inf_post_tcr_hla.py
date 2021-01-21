from DeepTCR.DeepTCR import DeepTCR_WF
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score,roc_curve
import glob

gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

DTCR = DeepTCR_WF('load')
DTCR.Get_Data(directory='../../Data_Post',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data_Post/HLA_Ref_sup_AB.csv')
beta_sequences = DTCR.beta_sequences
v_beta = DTCR.v_beta
d_beta = DTCR.d_beta
j_beta = DTCR.j_beta
hla = DTCR.hla_data_seq
sample_id = DTCR.sample_id
counts = DTCR.counts

DTCR = DeepTCR_WF('HLA_TCR')
DTCR.Sample_Inference(sample_labels=sample_id,beta_sequences=beta_sequences,v_beta=v_beta,d_beta=d_beta,j_beta=j_beta,
                      hla=hla,counts=counts)

files = glob.glob('../../Data_Post/crpr/*.tsv')
filescrpr = [os.path.basename(x) for x in files]
labelcrpr = [['crpr']*len(files)]

files = glob.glob('../../Data_Post/sdpd/*.tsv')
filessdpd = [os.path.basename(x) for x in files]
labelsdpd = [['sdpd']*len(files)]

files = np.hstack([filescrpr,filessdpd])
labels = np.squeeze(np.hstack([labelcrpr,labelsdpd]),0)
label_dict = dict(zip(files,labels))

df_pred = DTCR.Inference_Pred_Dict['crpr']
df_pred['label'] = df_pred['Samples'].map(label_dict)
df_pred['label_bin'] = 0.0
df_pred['label_bin'][df_pred['label']=='crpr']=1.0
roc_auc_score(df_pred['label_bin'],df_pred['Pred'])

df_pred = df_pred[['Samples','Pred','label_bin']]
df_pred = df_pred.rename(columns={'Pred':'y_pred','label_bin':'y_test'})
df_pred.to_csv('sample_tcr_hla_inf.csv',index=False)
