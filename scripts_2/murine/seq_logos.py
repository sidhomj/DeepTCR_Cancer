from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

DTCR_WF = DeepTCR_WF('Rudqvist_WF',device=0)
df_preds = pd.read_csv('preds.csv')
class_sel = 'Control'
df_preds.sort_values(by=class_sel,inplace=True,ascending=False)

num = 10
beta_sequences = np.array(df_preds['beta_sequences'])[0:num]
v_beta = np.array(df_preds['v_beta'])[0:num]
d_beta = np.array(df_preds['d_beta'])[0:num]
j_beta = np.array(df_preds['j_beta'])[0:num]

models = np.random.choice(range(100),25,replace=False)
models = np.array(['model_'+str(x) for x in models])
DTCR_WF.Residue_Sensitivity_Logo(beta_sequences=beta_sequences,v_beta=v_beta,d_beta=d_beta,j_beta=j_beta,
                                 class_sel=class_sel,background_color='white',Load_Prev_Data=True,
                                 models=models,figsize=(3,4),min_size=0.75)