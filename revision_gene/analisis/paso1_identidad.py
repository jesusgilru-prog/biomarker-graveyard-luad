"""Paso 1 (ACTA): auditoria de identidad molecular HPA <-> matrices. Solo lectura de datos."""
import pandas as pd, numpy as np, gzip, json
R='data/raw/'; P='data/processed/'
df=pd.read_csv(R+'hpa/proteinatlas.tsv',sep='\t',low_memory=False)
col='Cancer prognostics - Lung Adenocarcinoma (TCGA)'
h=df[['Gene','Ensembl',col]].dropna(subset=[col]).copy()
h['call']=h[col].str.extract(r'^(.*)\s+\(')[0]; h['p_hpa']=h[col].str.extract(r'\(([\d.eE+-]+)\)')[0].astype(float)
hd=h[h.call.str.contains('favorable')].copy()   # direccionales (incluye unfavorable)
hd['validated']=hd.call.str.startswith('validated')
print('direccionales:',len(hd),'validated',hd.validated.sum(),'potential',(~hd.validated).sum())
# matrices
cp=pd.read_csv(R+'cptac/LUAD_tumor_RNA-Seq_Expr_WashU_FPKM.tsv.gz',sep='\t',usecols=[0,1])
cp['ens']=cp.gene_id.str.split('.').str[0]
cps=pd.read_csv(P+'cptac_luad_fpkm.csv.gz',usecols=[0]).iloc[:,0]
tc=pd.read_csv(P+'tcga_luad_tpm_full.csv.gz',usecols=[0]).iloc[:,0]
gs=pd.read_csv(P+'gse68465_luad_expr.csv.gz',usecols=[0]).iloc[:,0]
print('filas simbolo duplicadas: cptac',cps.duplicated().sum(),'tcga',tc.duplicated().sum(),'gse',gs.duplicated().sum())
print('gse con ///:',gs.str.contains('///').sum())
panel=set(cps)&set(tc)&set(gs)
print('interseccion simbolos 3 matrices:',len(panel))
hd['en_panel_simbolo']=hd.Gene.isin(panel)
# identidad por Ensembl en CPTAC
cmap=cp.drop_duplicates('ens').set_index('ens').gene_name
hd['sym_cptac_por_ens']=hd.Ensembl.map(cmap)
hd['ens_en_cptac']=hd.Ensembl.isin(cmap.index)
hd['sym_coincide']=hd.sym_cptac_por_ens==hd.Gene
hd['ens_de_simbolo_cptac']=hd.Gene.map(cp.drop_duplicates('gene_name').set_index('gene_name').ens)
hd['ens_coincide']=hd.ens_de_simbolo_cptac==hd.Ensembl
sub=hd[hd.en_panel_simbolo]
print('en panel por simbolo:',len(sub))
print(' ens en CPTAC:',sub.ens_en_cptac.sum(),'| simbolo por ens coincide:',sub.sym_coincide.sum(),'| ens del simbolo CPTAC == ens HPA:',sub.ens_coincide.sum())
bad=sub[~sub.ens_coincide]
print(bad[['Gene','Ensembl','ens_de_simbolo_cptac','sym_cptac_por_ens','call','p_hpa']].to_string())
print('no en panel (primeros):',hd[~hd.en_panel_simbolo].Gene.head(15).tolist(), 'n=',(~hd.en_panel_simbolo).sum())
print('de los no-panel, Ensembl presente en CPTAC:',hd[~hd.en_panel_simbolo].ens_en_cptac.sum())
ren=hd[(~hd.en_panel_simbolo)&hd.ens_en_cptac][['Gene','Ensembl','sym_cptac_por_ens']]
print(ren.head(30).to_string())
hd.to_csv('revision_gene_nov/analisis/paso1_hpa_direccional_identidad.csv',index=False)
json.dump({'n_dir':len(hd),'validated':int(hd.validated.sum()),'potential':int((~hd.validated).sum()),'panel_simbolos':len(panel),'en_panel':int(hd.en_panel_simbolo.sum())},open('revision_gene_nov/analisis/paso1_resumen.json','w'),indent=1)
