import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_WF
import pickle
import pandas as pd
import numpy as np

DTCR_load = DeepTCR_WF('Human_TIL')
DTCR_load.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

df = pd.DataFrame()
df['beta'] = DTCR_load.beta_sequences
df['v_beta'] = DTCR_load.v_beta
df['d_beta'] = DTCR_load.d_beta
df['j_beta'] = DTCR_load.j_beta
df['hla'] = DTCR_load.hla_data_seq
df['crpr'] = predicted[:,0]
df['sdpd'] = predicted[:,1]
class_sel = 'crpr'
df.sort_values(by=class_sel,ascending=False,inplace=True)
DTCR = DeepTCR_WF('HLA_TCR')
num_seq = 50
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(df['beta'].iloc[0:num_seq]),
                              v_beta=np.array(df['v_beta'].iloc[0:num_seq]),
                              d_beta = np.array(df['d_beta'].iloc[0:num_seq]),
                              j_beta = np.array(df['j_beta'].iloc[0:num_seq]),
                              hla=np.array(df['hla'].iloc[0:num_seq]),
                              models = ['model_'+str(x) for x in np.random.choice(range(100),1,replace=False)],
                              figsize=(4,8),
                              background_color='black',
                              class_sel = class_sel,
                              min_size=0.5,
                              Load_Prev_Data=True)