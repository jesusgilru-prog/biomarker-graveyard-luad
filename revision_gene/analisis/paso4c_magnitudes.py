"""Paso 4 revisado: estimador = mediana por gen de a_g = 1 - mean(b1 orientado)/mean(b0 orientado) sobre los 92 reproducidos;
bootstrap de pacientes con los MISMOS indices para todos los genes en cada replica; nulo = modulos co-expresados de 13 genes SIN acoplamiento con el metagen;
comparadores nombrados (alveolar, inmune). Ejecutar con miniconda python."""
import sys, json, time, os, warnings; warnings.filterwarnings('ignore')
HERE='/home/jesus/biomarker_graveyard/revision_gene_nov/analisis'
sys.path.insert(0,HERE); sys.argv=['x','1']
from multiprocessing import Pool
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
import paso3_estratificacion as P
sys.path.insert(0,'/home/jesus/biomarker_graveyard/revision_gene_nov/run_v2/src')
import phase6_stage_age_sensitivity as F6
from scipy.stats import rankdata
B=int(os.environ.get('BOOT','200')); NNULL=int(os.environ.get('NNULL','100'))
h=pd.read_csv(HERE+'/paso3_corpus_genes.csv'); genes=h.Gene.tolist(); dirs=h['dir'].values; rep=h.replica.values
cm=F6.cptac_stage_age(); gm=F6.gse68465_stage_age()
def prep(X,y,meta):
    ids=[c for c in meta.index if c in X.columns]; pos=[list(X.columns).index(c) for c in ids]
    return dict(pos=np.array(pos),t=y[:,0][pos],e=y[:,1][pos],cov=meta.loc[ids,['stage_late','age']].values)
C=prep(P.Xc,P.yc,cm); G=prep(P.Xg,P.yg,gm)
XC=P.Xc.loc[genes].values; XG=P.Xg.loc[genes].values
def zs(v): s=v.std(); return (v-v.mean())/s if s>0 else v*0
def fit(t,e,cols,strata=None):
    d=pd.DataFrame(cols); d['t']=t; d['e']=e
    if strata is not None: d['site']=strata
    try: return float(CoxPHFitter(penalizer=1e-6).fit(d,'t','e',strata=(['site'] if strata is not None else None)).params_['x'])
    except Exception as ex:
        FAILS.append(str(ex)[:80]); return np.nan
def b01(coh,x,m,idx):
    x=x[coh['pos']]; m=m[coh['pos']]; t,e,cov=coh['t'],coh['e'],coh['cov']
    st=SITE_G[coh['pos']] if (coh is G and os.environ.get('STRAT')=='1') else None
    if idx is not None:
        x,m,t,e,cov=x[idx],m[idx],t[idx],e[idx],cov[idx]
        if st is not None: st=st[idx]
    base=dict(x=zs(x),s=cov[:,0],a=cov[:,1]); return fit(t,e,base,st),fit(t,e,dict(base,m=zs(m)),st)
STATE={}
FAILS=[]
import paso8_dependencia_centro as P8
SITE_G=P8.site
EXCL=[g for g in os.environ.get('EXCL','').split(',') if g]
def job(a):
    j,ic,ig,mc,mg=a   # mc,mg: score del modulo (array por cohorte completa)
    r=[]
    for coh,X,m,idx in ((C,XC,mc,ic),(G,XG,mg,ig)): r+=list(b01(coh,X[j],m,idx))
    return r
def summarize(res,idxs):
    r=res[idxs]; b0=dirs[idxs][:,None]*np.column_stack([r[:,0],r[:,2]]); b1=dirs[idxs][:,None]*np.column_stack([r[:,1],r[:,3]])
    a=1-np.nanmean(b1,1)/np.nanmean(b0,1); return float(np.nanmedian(a))
def run(pool,mc,mg,ic=None,ig=None,idxs=None):
    idxs=np.where(rep)[0] if idxs is None else idxs
    res=np.full((len(genes),4),np.nan); out=pool.map(job,[(j,ic,ig,mc,mg) for j in idxs],chunksize=2); res[idxs]=np.array(out); return summarize(res,idxs),res
if __name__=='__main__':
    t0=time.time(); rj=np.where(rep)[0]
    with Pool(22) as pool:
        est,res=run(pool,P.mc,P.mg); print('estimador puntual (92 reproducidos):',round(est,4),flush=True)
        out={'puntual':est}
        # bootstrap con indices compartidos
        bs=[]
        for b in range(B):
            r=np.random.default_rng(1000+b); ic=r.integers(0,len(C['t']),len(C['t'])); ig=r.integers(0,len(G['t']),len(G['t']))
            if os.environ.get('STRAT')=='1':
                sg=SITE_G[G['pos']]; ig=np.concatenate([r.choice(np.where(sg==s_)[0],(sg==s_).sum()) for s_ in np.unique(sg)])
            bs.append(run(pool,P.mc,P.mg,ic,ig)[0])
            if b%25==0: print('boot',b,round(time.time()-t0),flush=True)
        out['bootstrap_replicas']=[float(v) for v in bs]; out['n_fallos_ajuste']=len(FAILS); out['bootstrap_ic95']=[float(np.nanpercentile(bs,2.5)),float(np.nanpercentile(bs,97.5))]; out['B']=B
        json.dump(out,open(HERE+'/'+os.environ.get('TAG','paso4c')+'_resultado.json','w'),indent=1)
        # modulos nombrados
        def mod(gs): 
            gs=[g for g in gs if g in P.Xc.index and g in P.Xg.index and g not in EXCL]
            return P.z(P.Xc.loc[gs].values).mean(0),P.z(P.Xg.loc[gs].values).mean(0),gs
        for name,gs in {'alveolar':['SFTPB','SFTPC','SFTPA1','SFTPA2','NAPSA','AGER','HOPX','LAMP3','SFTPD','ABCA3','SLC34A2','EMP2','CLIC5'],
                        'immune':['PTPRC','CD3E','CD3D','CD2','CD8A','CD4','CXCR4','LCP2','CD53','CORO1A','IL7R','TRAC','CD48']}.items():
            a,b,g2=mod(gs); out['modulo_'+name]=run(pool,a,b)[0]; print(name,out['modulo_'+name],flush=True)
        # nulo: modulos sin acoplamiento con el metagen
        allg=[g for g in P.panel if g not in P.META and g not in set(genes)]
        Zc=P.z(P.Xc.loc[allg].values); Zg=P.z(P.Xg.loc[allg].values)
        rmc=np.array([np.corrcoef(rankdata(Zc[i]),rankdata(P.mc))[0,1] for i in range(len(allg))]); rmg=np.array([np.corrcoef(rankdata(Zg[i]),rankdata(P.mg))[0,1] for i in range(len(allg))])
        ok=np.where((np.abs(rmc)+np.abs(rmg))/2<0.3)[0]; print('genes candidatos sin acoplamiento con metagen:',len(ok),'de',len(allg),flush=True)
        Cor=(Zc[ok]@Zc[ok].T/Zc.shape[1]+Zg[ok]@Zg[ok].T/Zg.shape[1])/2
        def zr(R): return (R-R.mean(1,keepdims=True))/R.std(1,keepdims=True)
        R1=zr(rankdata(XC[rj],axis=1)); R2=zr(rankdata(XG[rj],axis=1)); nulls=[]; rng=np.random.default_rng(11)
        for sd in rng.permutation(len(ok))[:3000]:
            nb=np.argsort(-Cor[sd])[:13]; mc=Zc[ok][nb].mean(0); mg=Zg[ok][nb].mean(0)
            rm=abs(np.corrcoef(rankdata(mc),rankdata(P.mc))[0,1]); rg=abs(np.corrcoef(rankdata(mg),rankdata(P.mg))[0,1])
            if (rm+rg)/2>=0.3: continue
            a1=np.abs(R1@zr(rankdata(mc)[None,:])[0]/R1.shape[1]); a2=np.abs(R2@zr(rankdata(mg)[None,:])[0]/R2.shape[1])
            nulls.append((float(np.median((a1+a2)/2)),mc,mg,(rm+rg)/2))
            if len(nulls)>=NNULL*3: break
        # quedarse con los NNULL de mayor acoplamiento con los reproducidos
        nulls=sorted(nulls,key=lambda x:-x[0])[:NNULL]; print('modulos nulos: mediana |rho| con reproducidos',round(np.mean([n[0] for n in nulls]),3),'(metagen: %.3f); |rho| con metagen medio %.3f'%(np.median(h.absrho[rep]),np.mean([n[3] for n in nulls])),flush=True)
        sh=[run(pool,n[1],n[2])[0] for n in nulls]
        out['nulo']=dict(media=float(np.mean(sh)),p2_5=float(np.percentile(sh,2.5)),p97_5=float(np.percentile(sh,97.5)),n=len(sh),rho_reproducidos=float(np.mean([n[0] for n in nulls])),rho_metagen=float(np.mean([n[3] for n in nulls])))
    json.dump(out,open(HERE+'/'+os.environ.get('TAG','paso4c')+'_resultado.json','w'),indent=1); print(json.dumps(out,indent=1))
