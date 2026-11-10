import sys; sys.path.insert(0,"/home/jesus/biomarker_graveyard/revision_gene_nov/analisis")
import sys, numpy as np, pandas as pd, json
from scipy import stats as sps
sys.argv=['x','1']; 
import paso3_estratificacion as P   # reutiliza carga (B=1: no ejecuta main)
h=P.h.copy()
pc=[];pg=[];dc=[];dg=[]
for j in range(len(h)):
    a,b=P.stat(P.XC[j],P.yc); c,d=P.stat(P.XG[j],P.yg); pc.append(a);dc.append(b);pg.append(c);dg.append(d)
h['p_c']=pc;h['d_c']=dc;h['p_g']=pg;h['d_g']=dg
def one(p,d,hd): return np.where(d==hd,p/2,1-p/2)
h['po_c']=one(h.p_c.values,h.d_c.values,h['dir'].values); h['po_g']=one(h.p_g.values,h.d_g.values,h['dir'].values)
h['pmax']=np.maximum(h.po_c,h.po_g)
def bh(p,by=False):
    m=len(p);o=np.argsort(p);c=np.sum(1/np.arange(1,m+1)) if by else 1
    q=np.empty(m);run=1
    for i in range(m-1,-1,-1):
        run=min(run,p[o[i]]*m*c/(i+1));q[o[i]]=run
    return q
h['q_bh']=bh(h.pmax.values);h['q_by']=bh(h.pmax.values,True)
print('corpus',len(h),'| marginales p<0.05 y direccion HPA: CPTAC',int(((h.p_c<.05)&(h.d_c==h['dir'])).sum()),'GSE',int(((h.p_g<.05)&(h.d_g==h['dir'])).sum()),'ambas',int(((h.p_c<.05)&(h.d_c==h['dir'])&(h.p_g<.05)&(h.d_g==h['dir'])).sum()))
print('conjuncion parcial BH q<0.05:',int((h.q_bh<.05).sum()),'| BY:',int((h.q_by<.05).sum()))
print(h.sort_values('pmax').head(6)[['Gene','call','po_c','po_g','pmax','q_bh','q_by']].round(4).to_string())
h.to_csv('paso5_conjuncion.csv',index=False)
