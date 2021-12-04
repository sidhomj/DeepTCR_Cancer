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

df_pre = df[df['freq_pre']>0]
df_agg = df_pre.groupby(['sample','seq_type','gt']).agg({'delta':np.sum}).reset_index()

plt.figure()
g = sns.boxplot(data=df_agg,hue='gt',y='delta',x='seq_type',showfliers=False,order=order,showmeans=True)
plt.xlabel('')
plt.ylabel('Δ Frequency',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('delta_pre_sample.png',dpi=1200)

df_post = df[df['freq_post']>0]
df_agg = df_post.groupby(['sample','seq_type','gt']).agg({'delta':np.sum}).reset_index()

plt.figure()
pal = {'crpr':'royalblue','sdpd':'red'}
g = sns.boxplot(data=df_agg,hue='gt',y='delta',x='seq_type',showfliers=False,order=order,palette=pal)
plt.xlabel('')
plt.ylabel('Δ Frequency',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('delta_post_sample.png',dpi=1200)