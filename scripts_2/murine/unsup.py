from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF, DeepTCR_U
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import umap

df_preds = pd.read_csv('preds.csv')

DTCR = DeepTCR_U('Rudqvist_U',device=0)
DTCR.Get_Data(directory='../../Rudqvist',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)
DTCR.Train_VAE(accuracy_min=0.98)
X_2 = umap.UMAP().fit_transform(DTCR.features)

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = np.ndarray.flatten(ax)
for ii,cl in enumerate(DTCR.lb.classes_,0):
    ax[ii].scatter(X_2[:,0],X_2[:,1],c=np.array(df_preds[cl]),cmap='jet')
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])
    ax[ii].set_title(cl)

