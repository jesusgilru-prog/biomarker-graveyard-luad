"""Paso 8b: submuestreo SIN reemplazo (80% de pacientes por cohorte, B=500); quintiles fijos. Descriptivo: estabilidad de la direccion de la tendencia."""
import json, time
from multiprocessing import Pool
import numpy as np
import paso8_dependencia_centro as P8
def one(seed):
    r=np.random.default_rng(seed)
    ic=np.sort(r.choice(len(P8.yc),int(.8*len(P8.yc)),replace=False)); ig=np.sort(r.choice(len(P8.yg),int(.8*len(P8.yg)),replace=False))
    hh=P8.hits(ic,ig,False); q=P8.qrates(hh); return q.tolist(),P8.ca(hh),int(hh.sum())
if __name__=='__main__':
    t=time.time()
    with Pool(22) as p: R=p.map(one,range(5000,5500),chunksize=5)
    q=np.array([x[0] for x in R]); z=np.array([x[1] for x in R]); n=np.array([x[2] for x in R])
    o=dict(B=500,frac=0.8,n_repro_mediana=float(np.median(n)),n_repro_p2_5_97_5=[float(np.percentile(n,2.5)),float(np.percentile(n,97.5))],
      CA_z_mediana=float(np.median(z)),CA_z_p2_5_97_5=[float(np.percentile(z,2.5)),float(np.percentile(z,97.5))],prop_z_mayor_1_96=float(np.mean(z>1.96)),
      prop_q5_mayor_q1=float(np.mean(q[:,4]>q[:,0])),tasas_q_mediana=np.median(q,0).tolist(),tasas_q_p2_5=np.percentile(q,2.5,0).tolist(),tasas_q_p97_5=np.percentile(q,97.5,0).tolist())
    json.dump(o,open(P8.OUT/'paso8b_resultado.json','w'),indent=1); print(json.dumps(o,indent=1),round(time.time()-t))
