import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

df = pd.read_csv('CM038_BM.csv')

kmf = KaplanMeierFitter()
idx = df['DeepTCR'] >= np.median(df['DeepTCR'])
t = df['PFSINV']
c = df['PFSINV_CNSR']

kmf = KaplanMeierFitter()
kmf.fit(t[idx], 1 - c[idx],label='High (n= ' + str(np.sum(idx))+')')
ax = kmf.plot(show_censors=True, censor_styles={'ms': 4, 'marker': '+'},color='b',ci_show=False)
kmf.fit(t[~idx], 1 - c[~idx],label='Low (n = ' + str(np.sum(~idx))+')')
kmf.plot(ax=ax, show_censors=True, censor_styles={'ms': 4, 'marker': '+'},color='r',ci_show=False)
plt.xlim([0,24])
plt.xlabel('Months',fontdict={'size':14})
plt.ylabel('Progression Free Survival (PFS)',fontdict={'size':14})
r = logrank_test(t[idx], t[~idx], 1 - c[idx], 1 - c[~idx])
plt.text(16,0.05,"p = " + str(np.round(r.p_value,3)),fontdict={'family':'arial','size':24})
handles, labels = ax.get_legend_handles_labels()
handles = [handles[0],handles[1]]
labels = [labels[0],labels[1]]
leg = ax.legend(handles, labels,title='Likelihood of Response',frameon=False,fontsize=12,markerscale=4)
leg.get_title().set_fontsize('12')
leg._legend_box.align = "left"
plt.show()
check=1
