import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
from DeepTCR.functions.data_processing import Process_Seq
import numpy as np

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
# df_train['label'][df_train['label']=='Viral'] = 'Other'
# df_train['label'][df_train['label']=='NeoAg'] = 'Other'


beta_sequences = np.array(df_train['CDR3.beta.aa'])
class_labels = np.array(df_train['label'])
DTCR = DeepTCR_SS('mcpas')
DTCR.Load_Data(beta_sequences=beta_sequences,class_labels=class_labels)
DTCR.Monte_Carlo_CrossVal(folds=25,weight_by_class=True)

DTCR.AUC_Curve()
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr,tpr,th = roc_curve(DTCR.y_test[:,0],DTCR.y_pred[:,0])
plt.plot(th,fpr)
df_roc = pd.DataFrame()
df_roc['th'] = th
df_roc['fpr']=fpr
df_roc['tpr']=tpr


