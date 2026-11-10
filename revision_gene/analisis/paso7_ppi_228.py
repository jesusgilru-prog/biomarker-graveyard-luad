"""Test de direccion en la red STRING del panel de 121: todas las aristas y solo las de evidencia experimental/base de datos (escore>0 o dscore>0)."""
import os, pandas as pd, numpy as np
R='/home/jesus/biomarker_graveyard/'; H=os.path.dirname(os.path.abspath(__file__))
panel=pd.read_csv(R+'results/table_s1_121_gene_panel.csv'); d=dict(zip(panel.gene,panel.direction.str.startswith('unfavorable')))
ppi=pd.read_csv(R+'data/processed/string_ppi_panel121.tsv',sep='\t')
def test(e,name,N=5000):
    e=[(a,b) for a,b in zip(e.preferredName_A,e.preferredName_B) if a in d and b in d and a!=b]
    obs=sum(d[a]==d[b] for a,b in e); rng=np.random.default_rng(42); g=list(d); v=np.array(list(d.values())); cnt=[]
    for _ in range(N):
        s=dict(zip(g,rng.permutation(v))); cnt.append(sum(s[a]==s[b] for a,b in e))
    cnt=np.array(cnt); p=(np.sum(cnt>=obs)+1)/(N+1)
    return f'{name}: aristas={len(e)} misma direccion={obs} ({obs/len(e):.1%}) nulo media={cnt.mean():.1f} sd={cnt.std():.1f} p={p:.5f}'
out=[test(ppi,'todas (score combinado)'),test(ppi[(ppi.escore>0)|(ppi.dscore>0)],'solo escore>0 o dscore>0')]
open(H+'/paso7_ppi_228.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
