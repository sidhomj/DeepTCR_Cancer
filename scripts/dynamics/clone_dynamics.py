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
# df = df[(df['freq_pre']>0.01) & (df['freq_post']>0.00)]
# df = df[df['counts_pre']>10]
# df = df[df['freq_pre']>0]
df = df[df['freq_post']>0]
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
g = sns.boxplot(data=df,hue='gt',y='abs_delta',x='seq_type',showfliers=False,order=order,showmeans=True)
plt.xlabel('')
plt.ylabel('Absolute Δ',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('abs_delta.png',dpi=600)

df_s = df[df['sample'] == np.unique(df['sample'])[10]]
sns.boxplot(data=df_s,x='seq_type',y='abs_delta',showfliers=False)
plt.yscale('log')

df_crpr = df[df['gt']=='crpr']
# sns.scatterplot(data=df_crpr,x='pred',y='abs_delta')
x = np.array(df_crpr['pred'])
y = np.array(df_crpr['abs_delta'])
z = np.array(df_crpr['freq_pre'])
r = np.argsort(z)
plt.scatter(x[r],y[r],c=z[r],cmap='jet')
plt.yscale('log')

df_sdpd = df[df['gt']=='sdpd']
x = np.array(df_sdpd['pred'])
y = np.array(df_sdpd['abs_delta'])
z = np.array(df_sdpd['freq_pre'])
r = np.argsort(z)
plt.figure()
plt.scatter(x[r],y[r],c=z[r],cmap='jet')
plt.yscale('log')



sns.jointplot(data=df_crpr,x='pred',y='abs_delta')
plt.contour(df_crpr['pred'],df_crpr['abs_delta'],df_crpr['freq_pre'])

X, Y = np.meshgrid(x, y)



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