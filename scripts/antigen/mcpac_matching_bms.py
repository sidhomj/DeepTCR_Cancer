"""
Figure 5D,E
"""

"""
This scripts provides the visualization for the distribution of predictions for the McPas-TCR sequences, stratified by antigen cateogory. As well as creating the umaps showing where they lie in the unsupervised sequence space shown in Figure 2.

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DeepTCR.functions.data_processing import Process_Seq
from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF
import pickle
import os
from scipy.stats import gaussian_kde
from matplotlib.patches import Circle

def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r

df = pd.read_csv('../../Data/other/McPAS-TCR.csv')
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
cat_dict['Melan-A/MART-1'] = 'MART-1'
cat_dict['Melan-A A27L'] = 'MART-1'
cat_dict['MelanA/MART1'] = 'MART-1'
cat_dict['GP100-IMD'] = 'GP100'
cat_dict['mutated CDKNA2'] = 'CDKNA2'
cat_dict['BAGE'] = 'BAGE'
cat_dict['MAGE-1'] = 'MAGE'
cat_dict['gp100'] = 'GP100'
df_maa = df_maa.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first'}).reset_index()
df_maa['label'] = df_maa['Antigen.protein'].map(cat_dict)
df_maa['label2'] = df_maa['label']
# df_maa['label'] = 'MAA'
train_list.append(df_maa)

df_path = df[df['Category']=='Pathogens']
path_sel = ['Influenza','Cytomegalovirus (CMV)','Epstein Barr virus (EBV)','Yellow fever virus']
df_path = df_path[df_path['Pathology'].isin(path_sel)]
df_path = df_path.groupby(['CDR3.beta.aa']).agg({'Antigen.protein':'first','Epitope.peptide':'first','Pathology':'first'}).reset_index()
# df_path['label'] = 'Viral'
df_path['label'] = df_path['Pathology']
df_path['label2'] = 'Viral'
label_dict = {'Melan-A/MART-1':'MART-1',
              'Cytomegalovirus (CMV)':'CMV',
              'Influenza':'Flu',
              'Epstein Barr virus (EBV)':'EBV',
              'Yellow fever virus':'YF'}
path_sel = list(map(label_dict.get,path_sel))
df_path['label'] = df_path['label'].map(label_dict)
train_list.append(df_path)

df_train = pd.concat(train_list)
df_train = Process_Seq(df_train,'CDR3.beta.aa')

DTCR = DeepTCR_SS('../models/HLA_TCR')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('../models/cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

file = '../unsup/cm038_x2_u.pkl'
featurize = False
with open(os.path.join(file),'rb') as f:
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
color_tumor = 'darkorange'
color_viral = 'grey'
pal = {'MART-1':color_tumor,'CMV':color_viral, 'Flu':color_viral,
       'EBV':color_viral, 'YF':color_viral}
fig,ax = plt.subplots(figsize=(len(order)*1.25,5))
sns.violinplot(data=df_merge,x='label',y='pred',cut=0,order=order,palette=pal,ax=ax)
plt.ylabel('P(Response)',fontsize=24)
plt.xlabel('')
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.ylim([0,1])
plt.tight_layout()
plt.savefig('mcpas_violin.png',dpi=1200)
df_merge.sort_values(by='label',inplace=True)

win = 10
cut_bottom = np.percentile(predicted[:,0],win)
cut_top = np.percentile(predicted[:,0],100-win)
df_merge = df_merge[(df_merge['pred'] > cut_top) | (df_merge['pred']< cut_bottom)]

#umap
# fig,ax = plt.subplots(1,len(order),figsize=(14,4))
# for ii,l in enumerate(order):
#     df_plot = df_merge[df_merge['label']==l]
#     x,y,c,_,_ = GKDE(np.array(df_plot['x']),np.array(df_plot['y']))
#     # x,y,c = np.array(df_plot['x']),np.array(df_plot['y']),df_plot['pred']
#     ax[ii].scatter(x,y,c=c,cmap='jet',s=25)
#     ax[ii].set_title(l)
#     ax[ii].set_xticks([])
#     ax[ii].set_yticks([])
# plt.tight_layout()

plt.scatter(df_merge['x'],df_merge['y'])
plt.gca().set_aspect('equal')
xlim = plt.xlim()
ylim = plt.ylim()
dim1 = xlim[1]-xlim[0]
dim2 = ylim[1]-ylim[0]
maxdim = np.max([dim1,dim2])
r = (maxdim/2)*1.3
c_coord = (xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2
xlim = c_coord[0]-r,c_coord[0]+r
ylim = c_coord[1]-r,c_coord[1]+r
plt.xlim(xlim)
plt.ylim(ylim)
plt.close()
# c_coord = (xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2
# r = np.sqrt(np.square(xlim[1]-xlim[0])+np.square(ylim[1]-ylim[0]))/2

order = ['Viral','MART-1']
# fig,ax = plt.subplots(1,len(order),figsize=(len(order)*4,4))
fig,ax = plt.subplots(len(order),1,figsize=(4,len(order)*4))
for ii,l in enumerate(order):
    df_plot = df_merge[df_merge['label2']==l]
    x,y,c,_,_ = GKDE(np.array(df_plot['x']),np.array(df_plot['y']))
    # x,y,c = np.array(df_plot['x']),np.array(df_plot['y']),df_plot['pred']
    ax[ii].scatter(x,y,c=c,cmap='jet',s=25)
    ax[ii].set_title(l,fontsize=24)
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])
    ax[ii].set_xlim(xlim)
    ax[ii].set_ylim(ylim)
    ax[ii].add_artist(Circle(c_coord,r*0.95,color='grey',fill=False,lw=5))
    ax[ii].set_aspect('equal')
    ax[ii].axis('off')
plt.tight_layout()
plt.savefig('umap_matched.png',dpi=1200)
