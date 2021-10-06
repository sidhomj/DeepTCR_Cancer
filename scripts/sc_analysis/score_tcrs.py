from DeepTCR.DeepTCR import DeepTCR_WF
import pandas as pd
import numpy as np
from DeepTCR.functions.data_processing import supertype_conv_op
def process_gene(gene,chain,df_merge):
    temp = df_merge['TR'+chain+gene+'_1'].str.split('TR'+chain+gene,expand=True)[1].str.split('-',expand=True)
    temp.fillna(value=1,inplace=True)
    temp[0] = temp[0].astype(int).map("{:02}".format)
    temp[1] = temp[1].astype(int).map("{:02}".format)
    temp[2] = 'TCR'+chain + gene + temp[0].astype(str) + '-'+temp[1].astype(str)
    return temp[2]

df_tcr = pd.read_csv('../../SC/scTCRs.csv')
df_tcr.dropna(subset=['CDR3B_1'],inplace=True)
df_tcr['patient'] = df_tcr['patient'].str.split('.',expand=True)[0]

df_hla = pd.read_csv('../../SC/hla.csv')
df_merge = pd.merge(df_tcr,df_hla,on='patient')
df_merge['TRBV_1'] = process_gene('V','B',df_merge)
df_merge['TRBJ_1'] = process_gene('J','B',df_merge)
df_merge['TRBD_1'][df_merge['TRBD_1']=='None'] = 'nan'

beta_sequences = np.array(df_merge['CDR3B_1'])
v_beta = np.array(df_merge['TRBV_1'])
d_beta = np.array(df_merge['TRBD_1'])
j_beta = np.array(df_merge['TRBJ_1'])
hla = np.array(df_merge[['0','1','2','3','4','5']])
hla = np.array(supertype_conv_op(hla))
DTCR = DeepTCR_WF('../human/HLA_TCR')
out = DTCR.Sample_Inference(beta_sequences=beta_sequences,v_beta=v_beta,d_beta=d_beta,j_beta=j_beta,hla=hla)
                            # models=['model_'+str(x) for x in range(10)])
df_merge['pred'] = out[:,0]
df_merge.to_csv('tcrs_scored.csv',index=False)