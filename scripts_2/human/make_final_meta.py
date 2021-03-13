import pandas as pd
import numpy as np
from DeepTCR.DeepTCR import DeepTCR_WF

DTCR = DeepTCR_WF('load_data')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

df_master = pd.read_csv('Master_Beta.csv')
df_master.dropna(subset=['Pre_Sample'],inplace=True)
df_master = df_master[df_master['Pre_Sample'].isin(DTCR.sample_list)]
df_master.to_csv('Master_Beta_final.csv',index=False)