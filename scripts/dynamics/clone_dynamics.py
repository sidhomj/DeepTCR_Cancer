import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('df_dynamics.pkl','rb') as f:
    df = pickle.load(f)

df['pred'] = 1 - df['pred']
df['seq_type'] = None
cuts = list(range(0,110,10))
order = []
for ii,c in enumerate(cuts,0):
    try:
        sel = (df['pred'] >= cuts[ii]/100) & (df['pred'] < cuts[ii+1]/100)
        df['seq_type'][sel] = str(ii)
        order.append(str(ii))
    except:
        continue

plt.figure()
g = sns.boxplot(data=df,hue='gt',y='delta',x='seq_type',showfliers=False,order=order,showmeans=True)
plt.xlabel('')
plt.ylabel('Δ',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('delta.png',dpi=600)

plt.figure()
g = sns.boxplot(data=df,hue='gt',y='abs_delta',x='seq_type',showfliers=False,order=order,showmeans=True,whis=[5,95])
plt.xlabel('')
plt.ylabel('Absolute Δ',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('abs_delta.png',dpi=600)


# cut1 = np.percentile(df['pred'],90)
# cut2 = np.percentile(df['pred'],10)
#
# df['seq_type'] = None
# df['seq_type'][df['pred']>cut1] = 'tumor'
# df['seq_type'][df['pred']<cut2] = 'viral'
# df['seq_type'].fillna('other',inplace=True)
#
# sns.boxplot(data=df,hue='gt',y='abs_delta',x='seq_type',showfliers=False)
# plt.xlabel('')
# plt.ylabel('Absolute Delta')
# plt.tight_layout()