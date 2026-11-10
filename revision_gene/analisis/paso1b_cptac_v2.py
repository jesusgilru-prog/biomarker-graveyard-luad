"""Matriz CPTAC corregida: ante simbolos duplicados conserva la fila de mayor expresion media (no la primera)."""
import pandas as pd
raw=pd.read_csv('data/raw/cptac/LUAD_tumor_RNA-Seq_Expr_WashU_FPKM.tsv.gz',sep='\t').dropna(subset=['gene_name'])
raw['_m']=raw.iloc[:,2:].astype(float).mean(axis=1)
raw=raw.sort_values('_m',ascending=False).drop_duplicates('gene_name',keep='first')
v1=pd.read_csv('data/processed/cptac_luad_fpkm.csv.gz',index_col=0)
x=raw.set_index('gene_name').drop(columns=['gene_id','_m'])
x.columns=[c.replace('-T','') for c in x.columns]
x=x.loc[:,v1.columns].loc[v1.index]          # mismas muestras y mismo orden de genes
x.to_csv('data/processed/cptac_luad_fpkm_v2.csv.gz',compression='gzip')
d=(x.values!=v1.values).any(axis=1)
print('genes con fila cambiada:',d.sum(),'de',len(x)); print(list(x.index[d][:15]))
