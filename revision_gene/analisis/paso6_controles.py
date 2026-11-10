"""Paso 6 (tras revision cruzada): sensibilidades y controles. Ejecutar con miniconda python."""
import sys, json, warnings; warnings.filterwarnings('ignore')
HERE='/home/jesus/biomarker_graveyard/revision_gene_nov/analisis'
sys.path.insert(0,HERE); sys.argv=['x','1']
import numpy as np, pandas as pd
from scipy import stats
import paso3_estratificacion as P
h=pd.read_csv(HERE+'/paso3_corpus_genes.csv')
genes=h.Gene.tolist()
def zs(X): return (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
Xt=P.Xt
def rho_vs(X,m,gs): 
    from scipy.stats import rankdata
    R=rankdata(X.loc[gs].values,axis=1); R=(R-R.mean(1,keepdims=True))/R.std(1,keepdims=True)
    mm=rankdata(m); mm=(mm-mm.mean())/mm.std(); return R@mm/R.shape[1]
def trend(rep,absr,k):
    q=pd.qcut(absr,k,labels=False); g=pd.DataFrame({'q':q,'r':rep}).groupby('q').r.agg(['sum','count'])
    s=np.arange(k);N=g['count'].sum();R=g['sum'].sum();pb=R/N
    Tn=(s*(g['sum']-g['count']*pb)).sum();V=pb*(1-pb)*((g['count']*s**2).sum()-(g['count']*s).sum()**2/N)
    z=Tn/np.sqrt(V); return g['sum'].tolist(),g['count'].tolist(),float(z),float(2*stats.norm.sf(abs(z)))
rep=h.replica.values.astype(int)
out={}
# 1) |rho| medido solo en TCGA
mt=zs(Xt.loc[P.META].values).mean(0)
rt=np.abs(rho_vs(Xt,mt,genes)); out['rho_TCGA_quintiles']=trend(rep,rt,5); out['rho_TCGA_deciles']=trend(rep,rt,10)
# 2) terciles y quintiles originales
out['terciles']=trend(rep,h.absrho.values,3)
# 3) modulos comparadores (a posteriori, declarados): diferenciacion alveolar y linfocitos
MODS={'alveolar':['SFTPB','SFTPC','SFTPA1','SFTPA2','NAPSA','AGER','HOPX','LAMP3','SFTPD','ABCA3','SLC34A2','EMP2','CLIC5'],
      'immune':['PTPRC','CD3E','CD3D','CD2','CD8A','CD4','CXCR4','LCP2','CD53','CORO1A','IL7R','TRAC','CD48']}
for name,gs in MODS.items():
    gs=[g for g in gs if g in P.Xc.index and g in P.Xg.index]
    mc=zs(P.Xc.loc[gs].values).mean(0); mg=zs(P.Xg.loc[gs].values).mean(0)
    keep=~h.Gene.isin(gs).values
    a=(np.abs(rho_vs(P.Xc,mc,genes))+np.abs(rho_vs(P.Xg,mg,genes)))/2
    r=trend(rep[keep],a[keep],5); out['modulo_'+name]=dict(genes=gs,n_corpus=int(keep.sum()),quintiles=r)
    # correlacion del modulo con el metagen
    out['modulo_'+name]['rho_con_metagen_CPTAC']=float(stats.spearmanr(mc,P.mc)[0]); out['modulo_'+name]['rho_con_metagen_GSE']=float(stats.spearmanr(mg,P.mg)[0])
# 4) metagen: sus 12 genes en el corpus original (HPA) y significacion del metagen

# 4) genes con y sin llamada HPA, misma definicion libre de direccion HPA
from survival_stats import _logrank_components
allg=[g for g in P.panel if g not in P.META]
def pd_(X,y,gs):
    V=X.loc[gs].values; p=np.empty(len(gs)); d=np.empty(len(gs),int)
    for j in range(len(gs)):
        c,o=_logrank_components(V[j],y); p[j]=stats.chi2.sf(c,1); d[j]=1 if o>0 else -1 if o<0 else 0
    return p,d
pc,dc=pd_(P.Xc,P.yc,allg); pg,dg=pd_(P.Xg,P.yg,allg)
agree=(pc<.05)&(pg<.05)&(dc==dg)&(dc!=0)
absr=(np.abs(rho_vs(P.Xc,P.mc,allg))+np.abs(rho_vs(P.Xg,P.mg,allg)))/2
D=pd.DataFrame({'g':allg,'agree':agree,'absr':absr}); D['called']=D.g.isin(genes)
D=D[~D.g.isin(['MKKS'])]
qb=np.quantile(D.absr[D.called],[0.2,0.4,0.6,0.8])   # cortes de los quintiles del corpus
D['q']=np.digitize(D.absr,qb)
tab=D.groupby(['called','q']).agree.agg(['sum','count']); tab['rate']=tab['sum']/tab['count']; print(tab)
out['llamados_vs_no']=tab.reset_index().to_dict('records')
exp=0
for q in range(5):
    nc_=int(tab.loc[(True,q),'count']); ru=tab.loc[(False,q),'rate']; exp+=nc_*ru
obs=int(tab.loc[True,'sum'].sum()); out['exceso']=dict(observado_libre_direccion=obs,esperado_solo_acoplamiento=float(exp),agree_total_llamados=obs)
print('observado (libre de direccion HPA):',obs,'esperado por acoplamiento (genes sin llamada):',round(exp,1))
json.dump(out,open(HERE+'/paso6_controles.json','w'),indent=1,default=str)
for k,v in out.items():
    if k.startswith('modulo_') : print(k,v['genes'][:4],'...',v['quintiles'],'rho metagen',round(v['rho_con_metagen_CPTAC'],2),round(v['rho_con_metagen_GSE'],2))
print('TCGA-only rho quintiles',out['rho_TCGA_quintiles']); print('terciles',out['terciles'])
