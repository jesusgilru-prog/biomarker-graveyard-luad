"""Paso 3 (PROTOCOLO_CONGELADO.md): tasa de reproduccion de llamadas HPA por quintil de |rho| con el eje de proliferacion."""
import sys, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps
REPO=Path('/home/jesus/biomarker_graveyard'); sys.path.insert(0,str(REPO/'src'))
from survival_stats import _logrank_components
OUT=REPO/'revision_gene_nov/analisis'
META=['MKI67','TOP2A','PCNA','CCNB1','AURKA','TYMS','BIRC5','CDK1','MCM2','CCNA2','KIF20A','PLK1','TPX2']
B=int(sys.argv[1]) if len(sys.argv)>1 else 1000
def log2p1(d): return np.log2(d.clip(lower=0)+1)
Xc=log2p1(pd.read_csv(REPO/'data/processed/cptac_luad_fpkm_v2.csv.gz',index_col=0))
cc=pd.read_csv(REPO/'data/processed/cptac_luad_clinical.csv').set_index('case_id').reindex(Xc.columns)
Xg=log2p1(pd.read_csv(REPO/'data/processed/gse68465_luad_expr.csv.gz',index_col=0))
cg=pd.read_csv(REPO/'data/processed/gse68465_luad_clinical.csv').set_index('sample_id').reindex(Xg.columns)
Xt=log2p1(pd.read_csv(REPO/'data/processed/tcga_luad_tpm_full.csv.gz',index_col=0))
panel=sorted(set(Xc.index)&set(Xt.index)&set(Xg.index))
yc=cc[['os_time_days','os_event']].values.astype(float); yg=cg[['os_time_months','os_event']].values.astype(float)
# corpus con identidad Ensembl
h=pd.read_csv(OUT/'paso1_hpa_direccional_identidad.csv')
raw=pd.read_csv(REPO/'data/raw/cptac/LUAD_tumor_RNA-Seq_Expr_WashU_FPKM.tsv.gz',sep='\t')
raw['m']=raw.iloc[:,2:].astype(float).mean(axis=1); raw=raw.dropna(subset=['gene_name']).sort_values('m',ascending=False).drop_duplicates('gene_name')
ens_cptac=raw.set_index('gene_name').gene_id.str.split('.').str[0]
h=h[h.Gene.isin(panel)].copy()
h['ens_ok']=h.Gene.map(ens_cptac)==h.Ensembl
excl_id=h[~h.ens_ok].Gene.tolist()
h=h[h.ens_ok & ~h.Gene.isin(META)].copy()
h['dir']=np.where(h.call.str.contains('unfavorable'),1,-1)
genes=h.Gene.tolist()
print('corpus evaluable:',len(genes),'| excluidos por identidad:',excl_id,'| metagen en corpus excluidos:',int(h.shape[0]*0+len(set(META)&set(pd.read_csv(OUT/'paso1_hpa_direccional_identidad.csv').Gene))))
def z(x): return (x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)
def meta(X): return z(X.loc[META].values).mean(0)
mc,mg=meta(Xc),meta(Xg)
def rho(X,m,gs): return np.array([sps.spearmanr(X.loc[g].values,m)[0] for g in gs])
r_c=rho(Xc,mc,genes); r_g=rho(Xg,mg,genes)
h['rho_c']=r_c; h['rho_g']=r_g; h['absrho']=(np.abs(r_c)+np.abs(r_g))/2
XC=Xc.loc[genes].values; XG=Xg.loc[genes].values
def stat(x,y):
    chi,o=_logrank_components(x,y); return sps.chi2.sf(chi,1), (1 if o>0 else -1 if o<0 else 0)
def hits(yc_,yg_):
    out=np.zeros(len(genes),bool)
    for j,d in enumerate(h['dir'].values):
        p,dd=stat(XC[j],yc_)
        if dd!=d or p>=0.05: continue
        p,dd=stat(XG[j],yg_)
        if dd==d and p<0.05: out[j]=True
    return out
def one(seed):
    r=np.random.default_rng(seed); return hits(yc[r.permutation(len(yc))],yg[r.permutation(len(yg))])
if __name__=='__main__':
    t=time.time(); obs=hits(yc,yg); h['replica']=obs
    print('observado: reproducen en ambas',obs.sum(),'/',len(genes),f'({time.time()-t:.0f}s)')
    with Pool(20) as p: H=np.array(p.map(one,range(1000,1000+B),chunksize=5))
    print('perm hecho',time.time()-t)
    h['q']=pd.qcut(h.absrho,5,labels=False)
    rows=[]
    for q in range(5):
        m=(h.q==q).values; k=int(obs[m].sum()); n=int(m.sum())
        nullr=H[:,m].sum(1)/n
        lo,hi=np.percentile(nullr,[2.5,97.5])
        ci=sps.beta.interval(0.95,k+0.5,n-k+0.5)
        rows.append(dict(quintil=q+1,n=n,absrho_min=h.absrho[m].min(),absrho_max=h.absrho[m].max(),repro=k,tasa=k/n,ic95_lo=ci[0],ic95_hi=ci[1],nulo_media=nullr.mean(),nulo_p975=hi,p_emp=(np.sum(H[:,m].sum(1)>=k)+1)/(B+1)))
    T=pd.DataFrame(rows); print(T.round(4).to_string())
    # Cochran-Armitage sobre deciles
    h['dec']=pd.qcut(h.absrho,10,labels=False); g=h.groupby('dec').replica.agg(['sum','count'])
    s=np.arange(10); N=g['count'].sum(); R=g['sum'].sum(); pbar=R/N
    Tn=(s*(g['sum']-g['count']*pbar)).sum(); V=pbar*(1-pbar)*((g['count']*s**2).sum()-(g['count']*s).sum()**2/N)
    zca=Tn/np.sqrt(V); pca=2*sps.norm.sf(abs(zca))
    print('Cochran-Armitage deciles: z=%.3f p=%.3g'%(zca,pca))
    T.to_csv(OUT/'paso3_quintiles.csv',index=False); h.to_csv(OUT/'paso3_corpus_genes.csv',index=False)
    json.dump(dict(n_corpus=len(genes),excl_identidad=excl_id,observado=int(obs.sum()),B=B,CA_z=float(zca),CA_p=float(pca)),open(OUT/'paso3_resumen.json','w'),indent=1)
