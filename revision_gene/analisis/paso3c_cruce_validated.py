"""Cruce 5x5 (quintil de p_HPA x quintil de |rho|) y comparacion validated/potential. Lee paso3_corpus_genes.csv."""
import os, pandas as pd, numpy as np
from scipy import stats
H=os.path.dirname(os.path.abspath(__file__)); h=pd.read_csv(H+'/paso3_corpus_genes.csv')
h['pq']=pd.qcut(h.p_hpa.rank(method='first'),5,labels=False)
M=h.pivot_table(index='pq',columns='q',values='replica',aggfunc='mean')*100; N=h.pivot_table(index='pq',columns='q',values='replica',aggfunc='count')
M.to_csv(H+'/paso3_cruce5x5_tasa.csv'); N.to_csv(H+'/paso3_cruce5x5_n.csv')
h['val']=h.call.str.startswith('validated'); t=pd.crosstab(h.val,h.replica)
with open(H+'/paso3b_validated_vs_potential.txt','w') as f:
    for v,g in h.groupby('val'): print('validated' if v else 'potential',len(g),int(g.replica.sum()),round(g.replica.mean(),3),file=f)
    print('Fisher p=',stats.fisher_exact(t)[1],file=f); print(h.groupby(['val','q']).replica.mean().unstack().round(3),file=f)
print(M.round(1)); print(open(H+'/paso3b_validated_vs_potential.txt').read())
