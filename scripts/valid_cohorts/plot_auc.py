import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import distinctipy

df_yost = pd.read_csv('yost_preds.csv')
df_yost['cohort'] = 'yost'
df_sade = pd.read_csv('sade_preds.csv')
df_sade['cohort'] = 'sade'

df_preds = pd.concat([df_yost,df_sade])

n_boots = 5000
scores = []
name_list = []
names = ['yost','sade','yost+sade']
for name,df in zip(names,[df_yost,df_sade,df_preds]):
    for _ in range(n_boots):
        try:
            df_temp = df.sample(len(df),replace=True)
            scores.append(roc_auc_score(df_temp['response_bin'], df_temp['Pred']))
            name_list.append(name)
        except:
            continue

df_bs = pd.DataFrame()
df_bs['cohort'] = name_list
df_bs['auc'] = scores

for _ in range(n_boots):
    try:
        df_temp = df_preds.sample(len(df_preds),replace=True)
        scores.append(roc_auc_score(df_temp['response_bin'], df_temp['Pred']))
    except:
        continue


fig,ax = plt.subplots()
RGB_tuples = distinctipy.get_colors(len(names),rng=0)
color_dict = dict(zip(names,RGB_tuples))
for name,df in zip(names,[df_yost,df_sade,df_preds]):
    score = roc_auc_score(df['response_bin'],df['Pred'])
    fpr,tpr,_ = roc_curve(df['response_bin'],df['Pred'])
    key = name
    ax.plot(fpr, tpr, lw=2, label='%s (%0.2f)' % (key, score),color=color_dict[name])
ax.set_xlabel('False Positive Rate',fontsize=18)
ax.set_ylabel('True Positive Rate',fontsize=18)
ax.set_xlim([0,1])
ax.set_ylim([0,1.05])
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc='lower right')

ax2 = fig.add_axes([0.64, .28, .25, .25])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
sns.violinplot(data=df_bs,x='cohort',y='auc',ax=ax2,cut=0,palette=color_dict)
ax2.set_xlabel('')
ax2.hlines(y=0.5,xmin=ax2.get_xlim()[0],xmax=ax2.get_xlim()[1],linewidth=2,linestyles='--',color='navy')
ax2.set_xticks([])
# ax2.set_ylabel('')
fig.savefig('auc_sade_yost.png',dpi=1200)
percentileofscore(df_bs[df_bs['cohort']=='all']['auc'],0.5)
