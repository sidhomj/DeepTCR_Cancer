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

# df = df[(df['freq_pre'] > 0.001) | (df['freq_post']>.001)]
df = df[df['freq_post']>0]
df['fc_w'] = df['fc']*df['freq_pre']
df['delta_w'] = df['delta']*df['freq_pre']
df['abs_delta_w'] = df['abs_delta']*df['freq_pre']
df_agg = df.groupby(['sample','seq_type','gt']).agg({'delta':np.sum,'abs_delta':np.sum,
                                                     'freq_pre':np.sum,'freq_post':np.sum,
                                                     'fc':'mean','fc_w':'sum','abs_delta_w':'sum',
                                                     'delta_w':'sum'}).reset_index()
# df_agg['abs_delta_w'] = np.abs(df_agg['delta_w'])
# df_agg['abs_delta_norm'] = df_agg['abs_delta']/df_agg['freq_pre']

plt.figure()
g = sns.boxplot(data=df_agg,hue='gt',y='delta',x='seq_type',showfliers=False,order=order,showmeans=True)
plt.xlabel('')
plt.ylabel('Δ',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('sample_delta.png',dpi=1200)

plt.figure()
pal = {'crpr':'royalblue','sdpd':'red'}
g = sns.boxplot(data=df_agg,hue='gt',y='fc_w',x='seq_type',showfliers=False,order=order,palette=pal)
# g = sns.swarmplot(data=df_agg,hue='gt',y='abs_delta',x='seq_type',order=order)
plt.xlabel('')
plt.ylabel('Absolute Δ',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('sample_abs_delta.png',dpi=1200)