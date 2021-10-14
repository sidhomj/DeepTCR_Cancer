import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF
from DeepTCR.functions.data_processing import Process_Seq
import numpy as np
import pickle
import seaborn as sns

df = pd.read_csv('McPAS-TCR.csv')
df = df[df['Species']=='Human']
df.dropna(subset=['CDR3.beta.aa','Category'],inplace=True)
df = df[df['Category'].isin(['Pathogens','Cancer'])]

train_list = []
df_neo = df[df['Pathology']=='Neoantigen']
df_neo = df_neo[df_neo['Epitope.peptide']!='GILGFVFTL']
df_neo = df_neo.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first'}).reset_index()
df_neo['label'] = 'NeoAg'
train_list.append(df_neo)

df_maa = df[df['Pathology']=='Melanoma']
df_maa = df_maa.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first'}).reset_index()
df_maa['label'] = 'MAA'
train_list.append(df_maa)

df_path = df[df['Category']=='Pathogens']
path_sel = ['Influenza','Cytomegalovirus (CMV)','Epstein Barr virus (EBV)','Yellow fever virus']
df_path = df_path[df_path['Pathology'].isin(path_sel)]
df_path = df_path.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first','Pathology':'first'}).reset_index()
df_path['label'] = 'Viral'
# df_path['label'] = df_path['Pathology']
train_list.append(df_path)

df_train = pd.concat(train_list)
df_train = Process_Seq(df_train,'CDR3.beta.aa')


DTCR_load = DeepTCR_SS('Human_TIL')
DTCR_load.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')
#
with open('../human/cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

df_bms = pd.DataFrame()
df_bms['CDR3.beta.aa'] = DTCR_load.beta_sequences
df_bms['v_beta'] = DTCR_load.v_beta
df_bms['d_beta'] = DTCR_load.d_beta
df_bms['j_beta'] = DTCR_load.j_beta
df_bms['sample'] = DTCR_load.sample_id
df_bms['response'] = DTCR_load.class_id
df_bms['freq'] = DTCR_load.freq
df_bms['crpr'] = predicted[:,0]
df_bms['sdpd'] = predicted[:,1]
df_bms.sort_values(by='crpr',ascending=False,inplace=True)

df_merge = pd.merge(df_train,df_bms,on='CDR3.beta.aa')
df_agg = df_merge.groupby(['sample','label']).agg({'freq':'sum','response':'first'}).reset_index()
sns.violinplot(data=df_agg,x='label',y='freq',cut=0,hue='response',order=['MAA'])

df_agg = df_merge.groupby(['sample','label']).agg({'freq':'sum','response':'first'}).reset_index()
df_merge.sort_values(by='crpr',inplace=True,ascending=False)
df_merge = df_merge[df_merge['label']=='Viral']
df_agg = df_merge.groupby(['Antigen.protein']).agg({'crpr':'mean','Pathology':'first'}).reset_index()
sns.violinplot(data=df_agg,x='Pathology',y='crpr',cut=0)

df_agg = df_merge.groupby(['Epitope.peptide']).agg({'crpr':'mean'}).reset_index()
df_agg.sort_values(by='crpr',inplace=True,ascending=False)
DTCR = DeepTCR_SS('pep')
DTCR.Load_Data(beta_sequences=np.array(df_agg['Epitope.peptide']),Y=np.array(df_agg['crpr']))
DTCR.Monte_Carlo_CrossVal(folds=100)
DTCR.SRCC(kde=True)

df_merge.dropna(subset=['Epitope.peptide'],inplace=True)
df_merge.sort_values(by='crpr',inplace=True,ascending=False)
DTCR = DeepTCR_WF('pep_wf')
DTCR.Load_Data(beta_sequences=np.array(df_merge['Epitope.peptide']),sample_labels=np.array(df_merge['sample']),
               class_labels=np.array(df_merge['response']))
folds = 10
LOO = 6
epochs_min = 10
size_of_net = 'small'
num_concepts=64
hinge_loss_t = 0.3
train_loss_min=0.1
seeds = np.array(range(folds))
graph_seed = 0

DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,size_of_net=size_of_net, num_concepts=num_concepts,
                          combine_train_valid=True,hinge_loss_t=hinge_loss_t,train_loss_min=train_loss_min,seeds=seeds,
                          graph_seed=graph_seed)

DTCR = DeepTCR_SS('tcr_ss')
DTCR.Load_Data(beta_sequences=np.array(df_merge['CDR3.beta.aa']),Y=np.array(df_merge['crpr']))
DTCR.Monte_Carlo_CrossVal(folds=25)
DTCR.SRCC(kde=True)

df_bms = df_bms.sample(50000)
df_bms['bin'] = None
df_bms['bin'][df_bms['crpr'] > np.percentile(df_bms['crpr'],90)] = 'high'
df_bms['bin'][df_bms['crpr'] < np.percentile(df_bms['crpr'],10)] = 'low'
df_bms.dropna(subset=['bin'],inplace=True)

DTCR = DeepTCR_SS('tcr_ss')
# DTCR.Load_Data(beta_sequences=np.array(df_bms['CDR3.beta.aa']),Y=np.array(df_bms['crpr']))
DTCR.Load_Data(beta_sequences=np.array(df_bms['CDR3.beta.aa']),
               v_beta = np.array(df_bms['v_beta']),
               d_beta = np.array(df_bms['d_beta']),
               j_beta = np.array(df_bms['j_beta']),
               class_labels=np.array(df_bms['bin']))
DTCR.Monte_Carlo_CrossVal(folds=1,epochs_min=50)
DTCR.SRCC(kde=True)
DTCR.AUC_Curve()