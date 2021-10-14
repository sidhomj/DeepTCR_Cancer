import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

DTCR_load = DeepTCR_SS('Human_TIL')
DTCR_load.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('../human/cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

# DTCR = DeepTCR_SS('antigen_cat_beta')
# out = DTCR.Sequence_Inference(beta_sequences=DTCR_load.beta_sequences,v_beta=DTCR_load.v_beta,j_beta=DTCR_load.j_beta,
#                         hla=DTCR_load.hla_data_seq,models=['model_'+str(x) for x in np.random.choice(range(100),10,replace=False)])
# out = DTCR.Sequence_Inference(beta_sequences=DTCR_load.beta_sequences,v_beta=DTCR_load.v_beta,j_beta=DTCR_load.j_beta,
#                               hla=DTCR_load.hla_data_seq)
# out = DTCR.Sequence_Inference(beta_sequences=DTCR_load.beta_sequences)

DTCR = DeepTCR_SS('mcpas')
out = DTCR.Sequence_Inference(beta_sequences=DTCR_load.beta_sequences)

df = pd.DataFrame()
df['label'] = DTCR_load.class_id
df['sample'] = DTCR_load.sample_id
df['freq'] = DTCR_load.freq
for ii,c in enumerate(DTCR.lb.classes_):
    df[c] = out[:,ii]*df['freq']
df['crpr'] = predicted[:,0]
# df = df[(df['crpr']>np.percentile(df['crpr'],90)) | (df['crpr']<np.percentile(df['crpr'],10))]
df['crpr'] = df['crpr']*df['freq']
df = df[df['MAA']>0.60]
df = df.groupby(['sample','label']).sum().reset_index()
df.sort_values(by='label',inplace=True,ascending=False)
plot_c = []
plot_c.append(DTCR.lb.classes_)
# plot_c.append('crpr')
df = pd.melt(df,id_vars=['sample','label'],value_vars=np.hstack(plot_c))
sns.violinplot(data=df,x='variable',y='value',hue='label',cut=0)
sns.violinplot(data=df,x='variable',y='value',hue='label',cut=0,order=['MAA'])


from scipy.stats import spearmanr
corr = []
for ii,c in enumerate(DTCR.lb.classes_):
    co,_ = spearmanr(out[:,ii],predicted[:,0])
    corr.append(co)
