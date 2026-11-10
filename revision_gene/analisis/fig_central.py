"""Figura central (Fig. 2 del manuscrito). Lee paso3_quintiles.csv, paso3_cruce5x5_tasa.csv y paso4b_resultado.json."""
import os, json, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); os.makedirs(HERE+'/fig',exist_ok=True)
T=pd.read_csv(HERE+'/paso3_quintiles.csv'); R=json.load(open(HERE+'/paso4b_resultado.json')); M=pd.read_csv(HERE+'/paso3_cruce5x5_tasa.csv',index_col=0)
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,3,figsize=(14,4.4),gridspec_kw={'width_ratios':[1.15,1,0.95]})
x=np.arange(1,6); a=ax[0]
a.bar(x,T.tasa*100,color='#2b6cb0',width=.6)
a.errorbar(x,T.tasa*100,yerr=[(T.tasa-T.ic95_lo)*100,(T.ic95_hi-T.tasa)*100],fmt='none',ecolor='k',capsize=3,lw=1)
a.plot(x,T.nulo_media*100,'o',color='#c05621',label='permutation null (mean)'); a.set_xticks(x)
a.set_xticklabels([f"Q{i}\n{lo:.2f}–{hi:.2f}" for i,lo,hi in zip(x,T.absrho_min,T.absrho_max)],fontsize=9)
a.set_xlabel(r'Quintile of |$\rho$| with metagene'); a.set_ylabel('Reproduced in both cohorts (%)'); a.set_title('A  Reproduction by coupling',loc='left',fontsize=11,fontweight='bold'); a.legend(frameon=False,fontsize=9,loc='upper left')
for s in ('top','right'): a.spines[s].set_visible(False)
b=ax[1]; b.imshow(M.values,cmap='Blues',aspect='auto',vmin=0,vmax=40)
for i in range(5):
    for j in range(5): b.text(j,i,f"{M.values[i,j]:.0f}",ha='center',va='center',fontsize=10,color='w' if M.values[i,j]>22 else 'k')
b.set_xticks(range(5)); b.set_xticklabels([f'Q{i}' for i in range(1,6)]); b.set_yticks(range(5)); b.set_yticklabels([f'P{i}' for i in range(1,6)])
b.set_xlabel(r'Quintile of |$\rho$|'); b.set_ylabel('Quintile of HPA $p$ (P1 smallest)'); b.set_title('B  % reproduced',loc='left',fontsize=11,fontweight='bold')
c=ax[2]; ci=R['bootstrap_ic95']; n=R['nulo']
c.errorbar([0],[R['puntual']*100],yerr=[[ (R['puntual']-ci[0])*100],[(ci[1]-R['puntual'])*100]],fmt='s',color='#2f855a',capsize=4,ms=8)
c.plot([1],[R['modulo_alveolar']*100],'D',color='#6b46c1',ms=8); c.plot([2],[R['modulo_immune']*100],'D',color='#718096',ms=8)
c.errorbar([3],[n['media']*100],yerr=[[ (n['media']-n['p2_5'])*100],[(n['p97_5']-n['media'])*100]],fmt='o',color='#c05621',capsize=4,ms=8)
c.set_xlim(-.6,3.6); c.set_xticks([0,1,2,3]); c.set_xticklabels(['Prolif.\nmetagene','Alveolar\nmodule','Lymph.\nmodule','Random\nlow-coupling'],fontsize=9)
c.axhline(0,color='grey',lw=.6); c.set_ylabel('Median attenuation of log-HR (%)'); c.set_title('C  Adjustment, 92 genes',loc='left',fontsize=11,fontweight='bold')
for s in ('top','right'): c.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(HERE+'/fig/fig_central.pdf'); plt.savefig(HERE+'/fig/fig_central.png',dpi=200)
