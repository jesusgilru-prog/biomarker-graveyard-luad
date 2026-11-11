"""Paso 8 (revision tras Astra, 11-nov-2026): (A) nulo de permutacion para el estadistico de tendencia (preserva dependencia entre genes);
(B) bootstrap de pacientes (indices compartidos por todos los genes) -> IC para tasas por quintil y tendencia; (C) mismo analisis estratificando
GSE68465 por centro (log-rank estratificado, permutacion y bootstrap dentro de centro). Quintiles FIJOS del paso 3."""
import sys, json, time, gzip
sys.argv=['x','1']
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats as sps
import paso3_estratificacion as P
from survival_stats import _logrank_components
OUT=P.OUT
h=pd.read_csv(OUT/'paso3_corpus_genes.csv'); genes=h.Gene.tolist(); dirs=h['dir'].values
Q=pd.qcut(h.absrho,5,labels=False).values; D=pd.qcut(h.absrho,10,labels=False).values
XC=P.XC; XG=P.XG; yc=P.yc; yg=P.yg
L=gzip.open(P.REPO/'data/raw/geo/GSE68465_series_matrix.txt.gz','rt').readlines()
ids=[x.strip('"\n') for x in next(l for l in L if l.startswith('!Sample_geo_accession')).strip().split('\t')[1:]]
src=[x.strip('"\n') for x in next(l for l in L if l.startswith('!Sample_source_name')).strip().split('\t')[1:]]
site=pd.Series(src,index=ids).reindex(P.Xg.columns).values
SITES=[s for s in np.unique(site)]
def lr(x,y):
    c,o=_logrank_components(x,y); return c,o
def stat_plain(x,y):
    c,o=lr(x,y); return sps.chi2.sf(c,1),(1 if o>0 else -1 if o<0 else 0)
def stat_strat(x,y,st):
    oe=0.;V=0.
    for s in SITES:
        m=st==s; c,o=lr(x[m],y[m])
        if c>0: oe+=o; V+=o*o/c
    if V<=0: return 1.0,0
    return sps.chi2.sf(oe*oe/V,1),(1 if oe>0 else -1)
def hits(ic,ig,strat):
    out=np.zeros(len(genes),bool)
    yc_,yg_=yc[ic],yg[ig]; st=site[ig]
    for j,d in enumerate(dirs):
        p,dd=stat_plain(XC[j][ic],yc_)
        if dd!=d or p>=0.05: continue
        p,dd=stat_strat(XG[j][ig],yg_,st) if strat else stat_plain(XG[j][ig],yg_)
        if dd==d and p<0.05: out[j]=True
    return out
def ca(hit):
    g=pd.DataFrame({'d':D,'h':hit}).groupby('d').h.agg(['sum','count']); s=np.arange(10);N=g['count'].sum();pb=g['sum'].sum()/N
    T=(s*(g['sum']-g['count']*pb)).sum();V=pb*(1-pb)*((g['count']*s**2).sum()-(g['count']*s).sum()**2/N)
    return float(T/np.sqrt(V)) if V>0 else 0.0
def qrates(hit): return np.array([hit[Q==q].mean() for q in range(5)])
def perm_idx(r,strat):
    ic=r.permutation(len(yc)); ig=np.arange(len(yg))
    # permutar etiquetas (tiempo/evento) de GSE68465: dentro de centro si strat. Se implementa permutando los INDICES de las etiquetas.
    return ic
def run(args):
    kind,seed,strat=args; r=np.random.default_rng(seed)
    if kind=='perm':
        # permutar (tiempo,evento) conjuntamente; en strat, dentro de cada centro
        global yc,yg
        yc0,yg0=yc,yg
        yc=yc0[r.permutation(len(yc0))]
        if strat:
            yg1=yg0.copy()
            for s in SITES:
                m=np.where(site==s)[0]; yg1[m]=yg0[m[r.permutation(len(m))]]
            yg=yg1
        else: yg=yg0[r.permutation(len(yg0))]
        hh=hits(np.arange(len(yc)),np.arange(len(yg)),strat); yc,yg=yc0,yg0
    else:
        ic=r.integers(0,len(yc),len(yc))
        if strat:
            ig=np.concatenate([r.choice(np.where(site==s)[0],(site==s).sum()) for s in SITES])
        else: ig=r.integers(0,len(yg),len(yg))
        hh=hits(ic,ig,strat)
    return qrates(hh),ca(hh),int(hh.sum())
if __name__=='__main__':
    t0=time.time(); out={}
    allc=np.arange(len(yc)); allg=np.arange(len(yg))
    with Pool(22) as pool:
        for strat in (False,True):
            tag='estratificado_centro' if strat else 'sin_centro'
            obs=hits(allc,allg,strat); o=dict(n_repro=int(obs.sum()),tasas_q=qrates(obs).tolist(),n_q=[int(obs[Q==q].sum()) for q in range(5)],CA_z=ca(obs))
            print(tag,'observado',o,flush=True)
            PB=pool.map(run,[('perm',2000+b,strat) for b in range(1000)],chunksize=5)
            zp=np.array([x[1] for x in PB]); nq=np.array([x[0] for x in PB])
            o['perm']=dict(B=1000,CA_z_media=float(zp.mean()),CA_z_sd=float(zp.std()),CA_z_p975=float(np.percentile(zp,97.5)),p_emp_CA=float((np.sum(zp>=o['CA_z'])+1)/1001),
                           q1_nulo_media=float(nq[:,0].mean()),q1_p_emp=float((np.sum(nq[:,0]>=o['tasas_q'][0])+1)/1001),n_repro_nulo_media=float(np.mean([x[2] for x in PB])))
            print(tag,'perm',o['perm'],round(time.time()-t0),flush=True)
            BB=pool.map(run,[('boot',3000+b,strat) for b in range(500)],chunksize=5)
            bq=np.array([x[0] for x in BB]); bz=np.array([x[1] for x in BB]); bn=np.array([x[2] for x in BB])
            o['boot']=dict(B=500,tasas_q_ic95=[[float(a),float(b)] for a,b in np.percentile(bq,[2.5,97.5],axis=0).T],CA_z_ic95=[float(np.percentile(bz,2.5)),float(np.percentile(bz,97.5))],
                           n_repro_ic95=[float(np.percentile(bn,2.5)),float(np.percentile(bn,97.5))],dif_q5_q1_ic95=[float(np.percentile(bq[:,4]-bq[:,0],2.5)),float(np.percentile(bq[:,4]-bq[:,0],97.5))],
                           q5_mayor_q1=float(np.mean(bq[:,4]>bq[:,0])),monotona_no_decreciente=float(np.mean([(np.diff(r)>=0).all() for r in bq])))
            print(tag,'boot',o['boot'],round(time.time()-t0),flush=True)
            out[tag]=o
    json.dump(out,open(OUT/'paso8_resultado.json','w'),indent=1); print('FIN',round(time.time()-t0))
