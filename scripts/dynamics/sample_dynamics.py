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

df_agg = df.groupby(['sample','seq_type','gt']).agg({'delta':'sum','abs_delta':'sum'}).reset_index()

plt.figure()
g = sns.boxplot(data=df_agg,hue='gt',y='delta',x='seq_type',showfliers=False,order=order)
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
g = sns.boxplot(data=df_agg,hue='gt',y='abs_delta',x='seq_type',showfliers=False,order=order,palette=pal)
# g = sns.swarmplot(data=df_agg,hue='gt',y='abs_delta',x='seq_type',order=order)
plt.xlabel('')
plt.ylabel('Absolute Δ',fontsize=26)
plt.xticks([])
ax = plt.gca()
g.legend_.set_title(None)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig('sample_abs_delta.png',dpi=1200)