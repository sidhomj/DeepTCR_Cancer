import pandas as pd
import numpy as np
from DeepTCR.DeepTCR import DeepTCR_WF
import pickle
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

DTCR = DeepTCR_WF('HLA_TCR')
DTCR.Get_Data(directory='../../Data',Load_Prev_Data=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/HLA_Ref_sup_AB.csv')

with open('cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

file = 'cm038_x2_u.pkl'
with open(file,'rb') as f:
    features = pickle.load(f)

win = 10
cut_bottom = np.percentile(predicted[:,0],win)
cut_top = np.percentile(predicted[:,0],100-win)

sample_id = DTCR.sample_id
features_crpr = features[predicted[:,0] > cut_top]
dist_crpr = pdist(features_crpr)
features_sdpd = features[predicted[:,0] < cut_bottom]
dist_sdpd = pdist(features_sdpd)

plt.hist(dist_crpr,100,alpha=0.5)
plt.hist(dist_sdpd,100,alpha=0.5)

df_plot = pd.DataFrame(dist_crpr)
df_plot['label'] = 'crpr'

df_plot2 = pd.DataFrame(dist_sdpd)
df_plot2['label'] = 'sdpd'

df_plot = pd.concat([df_plot,df_plot2])
df_plot = df_plot.sample(10000)
sns.violinplot(data=df_plot,x='label',y=0)

df_pred = pd.DataFrame()
df_pred['predicted'] = predicted[:,0]
df_pred['freq'] = DTCR.freq
df_pred['predicted_w'] = df_pred['predicted']*df_pred['freq']
df_pred['label'] = DTCR.class_id
df_pred['sample'] = DTCR.sample_id
df_pred = df_pred.groupby(['sample']).agg({'label':'first','predicted_w':'sum'})
sns.violinplot(data=df_pred,x='label',y='predicted_w')

with open('cm038_ft_pred.pkl','rb') as f:
    features,predicted = pickle.load(f)

file = 'cm038_x2_u.pkl'
with open(file,'rb') as f:
    x2 = pickle.load(f)

win = 10
cut_bottom = np.percentile(predicted[:,0],win)
cut_top = np.percentile(predicted[:,0],100-win)

features_crpr = features[predicted[:,0] > cut_top]
dist_crpr = pdist(features_crpr)
features_sdpd = features[predicted[:,0] < cut_bottom]
dist_sdpd = pdist(features_sdpd)

x2_crpr = x2[predicted[:,0] > cut_top]
dist_crpr_x2 = pdist(x2_crpr)
x2_sdpd = x2[predicted[:,0] < cut_bottom]
dist_sdpd_x2 = pdist(x2_sdpd)

plt.scatter(dist_crpr,dist_crpr_x2)
from scipy.stats import spearmanr
spearmanr(dist_crpr,dist_crpr_x2)