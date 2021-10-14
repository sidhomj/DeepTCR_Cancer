import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DeepTCR.functions.data_processing import Process_Seq
from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF
import pickle
import os
from scipy.stats import gaussian_kde
def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r

df = pd.read_csv('McPAS-TCR.csv')
df = df[df['Species']=='Human']
df.dropna(subset=['CDR3.beta.aa','Category'],inplace=True)
df = df[df['Category'].isin(['Pathogens','Cancer'])]

train_list = []
df_neo = df[df['Pathology']=='Neoantigen']
df_neo = df_neo[df_neo['Epitope.peptide']!='GILGFVFTL']
df_neo = df_neo.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first'}).reset_index()
df_neo['label'] = 'NeoAg'
# train_list.append(df_neo)

df_maa = df[df['Pathology']=='Melanoma']
cat_sel = ['Melan-A/MART-1','Melan-A A27L','MelanA/MART1','mutated CDKNA2','GP100-IMD','BAGE','MAGE-1','gp100']
df_maa = df_maa[df_maa['Antigen.protein'].isin(cat_sel)]
cat_dict = {}
cat_dict['Melan-A/MART-1'] = 'Melan-A/MART-1'
cat_dict['Melan-A A27L'] = 'Melan-A/MART-1'
cat_dict['MelanA/MART1'] = 'Melan-A/MART-1'
cat_dict['GP100-IMD'] = 'GP100'
cat_dict['mutated CDKNA2'] = 'CDKNA2'
cat_dict['BAGE'] = 'BAGE'
cat_dict['MAGE-1'] = 'MAGE'
cat_dict['gp100'] = 'GP100'
df_maa = df_maa.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first'}).reset_index()
df_maa['label'] = df_maa['Antigen.protein'].map(cat_dict)
# df_maa['label'] = 'MAA'
train_list.append(df_maa)

df_path = df[df['Category']=='Pathogens']
path_sel = ['Influenza','Cytomegalovirus (CMV)','Epstein Barr virus (EBV)','Yellow fever virus']
df_path = df_path[df_path['Pathology'].isin(path_sel)]
df_path = df_path.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first','Pathology':'first'}).reset_index()
# df_path['label'] = 'Viral'
df_path['label'] = df_path['Pathology']
train_list.append(df_path)

df_train = pd.concat(train_list)
df_train = Process_Seq(df_train,'CDR3.beta.aa')

DTCR = DeepTCR_SS('HLA_TCR')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('../human/cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

file = 'cm038_x2_u.pkl'
featurize = False
with open(os.path.join('../human',file),'rb') as f:
    X_2 = pickle.load(f)

df_bms = pd.DataFrame()
df_bms['CDR3.beta.aa'] = DTCR.beta_sequences
df_bms['v_beta'] = DTCR.v_beta
df_bms['d_beta'] = DTCR.d_beta
df_bms['j_beta'] = DTCR.j_beta
df_bms['hla'] = DTCR.hla_data_seq
df_bms['freq'] = DTCR.freq
df_bms['sample'] = DTCR.sample_id
df_bms['class'] = DTCR.class_id
df_bms['x'] = X_2[:,0]
df_bms['y'] = X_2[:,1]
df_bms['pred'] = predicted[:,0]

df_merge = pd.merge(df_train,df_bms,on='CDR3.beta.aa')

order_viral = list(df_merge[df_merge['label'].isin(path_sel)].groupby(['label']).mean().sort_values(by='pred').index)
order_maa = list(df_merge[df_merge['label'].isin(np.unique(list(cat_dict.values())))].groupby(['label']).mean().sort_values(by='pred').index)
order_maa = [order_maa[0]]
order = np.hstack([order_maa,order_viral])
sns.violinplot(data=df_merge,x='label',y='pred',cut=0,order=order)
plt.ylabel('P(Response)',fontsize=24)
plt.xticks(rotation=90)
plt.tight_layout()
df_merge.sort_values(by='label',inplace=True)

df_merge = df_merge[(df_merge['pred'] > 0.95) | (df_merge['pred']<0.1)]

#umap
fig,ax = plt.subplots(1,len(order),figsize=(14,4))
for ii,l in enumerate(order):
    df_plot = df_merge[df_merge['label']==l]
    x,y,c,_,_ = GKDE(np.array(df_plot['x']),np.array(df_plot['y']))
    # x,y,c = np.array(df_plot['x']),np.array(df_plot['y']),df_plot['pred']
    ax[ii].scatter(x,y,c=c,cmap='jet',s=25)
    ax[ii].set_title(l)
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])
plt.tight_layout()