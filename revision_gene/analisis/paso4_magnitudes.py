"""Paso 4: M0 (gen+edad+estadio) vs M1 (+ metagen de proliferacion) sobre los MISMOS pacientes. Cambio de log-HR, IC bootstrap de pacientes, nulo de metagenes aleatorios emparejados."""
import sys, time, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/jesus/biomarker_graveyard/revision_gene_nov/run_v2/src'); sys.path.insert(0,'/home/jesus/biomarker_graveyard/revision_gene_nov/analisis')
from multiprocessing import Pool
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
sys.argv=['x','1']
import paso3_estratificacion as P
sys.path.insert(0,'/home/jesus/biomarker_graveyard/revision_gene_nov/run_v2/src')
import phase6_stage_age_sensitivity as F6
PILOTO=False
B=int(__import__('os').environ.get('BOOT','200')); NNULL=int(__import__('os').environ.get('NNULL','100'))
h=pd.read_csv('/home/jesus/biomarker_graveyard/revision_gene_nov/analisis/paso3_corpus_genes.csv')
genes=h.Gene.tolist(); dirs=h['dir'].values
cm=F6.cptac_stage_age(); gm=F6.gse68465_stage_age()
def prep(X,y,meta,m):
    ids=[c for c in meta.index if c in X.columns]
    pos=[list(X.columns).index(c) for c in ids]
    return dict(ids=ids,pos=pos,t=y[:,0][pos],e=y[:,1][pos],cov=meta.loc[ids,['stage_late','age']].values,meta=m[pos])
C=prep(P.Xc,P.yc,cm,P.mc); G=prep(P.Xg,P.yg,gm,P.mg)
XCg=P.Xc.loc[genes].values; XGg=P.Xg.loc[genes].values
print('n adj: CPTAC',len(C['ids']),'GSE',len(G['ids']),flush=True)
def zs(v): s=v.std(); return (v-v.mean())/s if s>0 else v*0
def fit(t,e,cols):
    d=pd.DataFrame(cols); d['t']=t; d['e']=e
    try:
        c=CoxPHFitter(penalizer=1e-6).fit(d,'t','e'); return float(c.params_['x'])
    except Exception: return np.nan
def b01(coh,xrow,idx=None,meta=None):
    pos=coh['pos']; x=xrow[pos]; m=coh['meta'] if meta is None else meta[pos]
    t,e,cov=coh['t'],coh['e'],coh['cov']
    if idx is not None: x,m,t,e,cov=x[idx],m[idx],t[idx],e[idx],cov[idx]
    base=dict(x=zs(x),s=cov[:,0],a=cov[:,1])
    b0=fit(t,e,base); b1=fit(t,e,dict(base,m=zs(m)))
    return b0,b1
def gene_job(a):
    j,seed,boot=a
    out=[]
    for coh,Xg in ((C,XCg),(G,XGg)):
        idx=None
        if boot:
            r=np.random.default_rng(seed*7919+j); idx=r.integers(0,len(coh['t']),len(coh['t']))
        out+=list(b01(coh,Xg[j],idx))
    return out
def delta_stats(res,mask):
    # res: n x 4 (b0c,b1c,b0g,b1g); orientado por HPA; Delta = (b0-b1)*dir ; media en ambas cohortes
    d=dirs[:,None]*np.column_stack([res[:,0]-res[:,1],res[:,2]-res[:,3]])
    b0=dirs[:,None]*np.column_stack([res[:,0],res[:,2]])
    m=mask
    return dict(delta_med=float(np.nanmedian(d[m])),b0_med=float(np.nanmedian(b0[m])),shrink=float(np.nanmedian(d[m])/np.nanmedian(b0[m])))
if __name__=='__main__':
    n=len(genes) if not PILOTO else 50
    t0=time.time()
    with Pool(22) as pool: res=np.array(pool.map(gene_job,[(j,0,False) for j in range(n)],chunksize=4))
    print('puntual',n,'genes',time.time()-t0,'s',flush=True)
    if PILOTO: sys.exit()
    rep=h.replica.values
    out={'puntual':{'replicadores':delta_stats(res,rep),'todos':delta_stats(res,np.ones(len(h),bool))}}
    for q in range(5): out['puntual'][f'q{q+1}']=delta_stats(res,(h.q==q).values)
    print(json.dumps(out['puntual'],indent=1),flush=True)
    # bootstrap de pacientes
    bs=[]
    for b in range(B):
        with Pool(22) as pool: rb=np.array(pool.map(gene_job,[(j,b+1,True) for j in np.where(rep)[0]],chunksize=2))
        full=np.full((len(h),4),np.nan); full[np.where(rep)[0]]=rb
        bs.append(delta_stats(full,rep)['shrink'])
        if b%20==0: print('boot',b,time.time()-t0,flush=True)
    out['bootstrap_shrink_replicadores']=dict(ic95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],B=B)
    json.dump(out,open('/home/jesus/biomarker_graveyard/revision_gene_nov/analisis/paso4_resultado.json','w'),indent=1)
    # nulo emparejado: modulos coherentes de 13 genes (semilla aleatoria + 12 vecinos mas coexpresados), mediana |rho| con replicadores similar
    from scipy.stats import rankdata
    rng=np.random.default_rng(7); allg=[g for g in P.panel if g not in P.META and g not in set(genes)]
    Zc=P.z(P.Xc.loc[allg].values); Zg=P.z(P.Xg.loc[allg].values)
    Cor=(Zc@Zc.T/Zc.shape[1]+Zg@Zg.T/Zg.shape[1])/2
    R1=rankdata(XCg[rep],axis=1); R2=rankdata(XGg[rep],axis=1)
    def zr(R): return (R-R.mean(1,keepdims=True))/R.std(1,keepdims=True)
    R1=zr(R1);R2=zr(R2)
    ref=np.median(h.absrho[rep]); nulls=[]; tries=0
    for sd in rng.permutation(len(allg))[:4000]:
        tries+=1; nb=np.argsort(-Cor[sd])[:13]
        mc_=Zc[nb].mean(0); mg_=Zg[nb].mean(0)
        r1=np.abs(R1@zr(rankdata(mc_)[None,:])[0]/R1.shape[1]); r2=np.abs(R2@zr(rankdata(mg_)[None,:])[0]/R2.shape[1])
        med=np.median((r1+r2)/2)
        if abs(med-ref)<=0.25*ref: nulls.append((nb,mc_,mg_,med))
        if len(nulls)>=NNULL: break
    print('nulos aceptados',len(nulls),'de',tries,'| ref |rho| mediana replicadores',round(ref,3),flush=True)
    sh=[]
    for s,mc_,mg_,med in nulls:
        Cn=dict(C,meta=mc_[:]); Gn=dict(G,meta=mg_[:])
        def job(j):
            r=[]
            for coh,Xg_,mm in ((C,XCg,mc_),(G,XGg,mg_)):
                r+=list(b01(coh,Xg_[j],None,mm))
            return r
        rr=np.array([job(j) for j in np.where(rep)[0]]); full=np.full((len(h),4),np.nan); full[np.where(rep)[0]]=rr
        sh.append(delta_stats(full,rep)['shrink'])
    out['nulo_emparejado_shrink']=dict(media=float(np.mean(sh)),p2_5=float(np.percentile(sh,2.5)),p97_5=float(np.percentile(sh,97.5)),n=len(sh))
    print(json.dumps({k:v for k,v in out.items() if k!='puntual'},indent=1))
    json.dump(out,open('/home/jesus/biomarker_graveyard/revision_gene_nov/analisis/paso4_resultado.json','w'),indent=1)
