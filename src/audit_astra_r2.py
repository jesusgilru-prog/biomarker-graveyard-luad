"""Read-only audit of original results; writes separate R2 diagnostics and cohort summaries.
Run with the biomarker environment. Does not rerun permutations or fit predictive models.
"""
from pathlib import Path
import hashlib, itertools, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'nullcal_pkg/src'))
from data import load_cptac_luad_discovery, load_tcga_luad_replication, load_gse68465_replication
from survival_stats import cox_score_chi2, logrank_chi2
from nullcal.corrections import benjamini_hochberg, benjamini_yekutieli
from phase6_stage_age_sensitivity import gse68465_stage_age
R=ROOT/'results'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(name,data):
    (R/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(name, json.dumps(data,allow_nan=False),flush=True)

# An independent exact null: enumerate all subsets of d deaths at each risk set.
def enumerated_score(x,y):
    u=v=0.
    for time in np.unique(y[y[:,1]==1,0]):
        risk=x[y[:,0]>=time]; dead=x[(y[:,0]==time)&(y[:,1]==1)]
        sums=np.array([sum(a) for a in itertools.combinations(risk,len(dead))])
        u+=dead.sum()-sums.mean(); v+=sums.var()
    return u*u/v if v else 0.

def breslow_score(x,y):
    u=v=0.
    for time in np.unique(y[y[:,1]==1,0]):
        risk=x[y[:,0]>=time]; dead=x[(y[:,0]==time)&(y[:,1]==1)]
        u+=dead.sum()-len(dead)*risk.mean(); v+=len(dead)*risk.var()
    return u*u/v if v else 0.

x=np.array([2.,-1.,3.,0.,5.,1.])
y=np.array([[1,1],[1,1],[1,0],[2,1],[3,0],[4,1]],dtype=float)
exact=enumerated_score(x,y); actual=cox_score_chi2(x,y); bres=breslow_score(x,y)
assert np.isclose(actual,exact) and not np.isclose(actual,bres)
y_no_ties=y.copy(); y_no_ties[:,0]=np.arange(1,7)
assert np.isclose(cox_score_chi2(x,y_no_ties),breslow_score(x,y_no_ties))
xb=np.array([0.,1.,0.,1.,1.,0.]); g=xb>0
ll=logrank_test(y[g,0],y[~g,0],y[g,1],y[~g,1]).test_statistic
assert np.isclose(cox_score_chi2(xb,y),ll)
assert np.isclose(logrank_chi2(xb,y),ll)
assert cox_score_chi2(np.ones(6),y)==0
rng=np.random.default_rng(42)
for i in range(30):
    xr=rng.normal(size=7); yr=np.column_stack([rng.integers(1,5,size=7),rng.integers(0,2,size=7)])
    assert np.isclose(cox_score_chi2(xr,yr),enumerated_score(xr,yr),rtol=1e-10,atol=1e-10)
print('PASS: exact-subset enumeration (31 examples), no-tie Breslow equivalence, binary lifelines log-rank equivalence, constant covariate.',flush=True)

cohorts={}
loaders={'cptac':load_cptac_luad_discovery,'tcga':load_tcga_luad_replication,'gse68465':load_gse68465_replication}
files={'cptac':'cptac_luad_clinical.csv','tcga':'tcga_luad_clinical_full.csv','gse68465':'gse68465_luad_clinical.csv'}
summary=json.loads((R/'phase6_stage_age_sensitivity.json').read_text())
cohort_summary={}
for tag,loader in loaders.items():
    X,clin=loader(); tc='os_time_months' if tag=='gse68465' else 'os_time_days'
    time=clin[tc].to_numpy(); event=clin.os_event.to_numpy(); yarr=np.column_stack([time,event]);cohorts[tag]=(X,yarr)
    km=KaplanMeierFitter().fit(time,1-event)
    vc=pd.Series(time[event==1]).value_counts()
    source=ROOT/'data'/'processed'/files[tag]
    cohort_summary[tag]={
      'n':len(clin),'events':int(event.sum()),'censored':int((1-event).sum()),
      'time_unit':'months' if tag=='gse68465' else 'days',
      'reverse_km_median_follow_up':float(km.median_survival_time_),
      'observed_time_median':float(np.median(time)),
      'observed_time_min':float(time.min()),'observed_time_max':float(time.max()),
      'missing_os_time':int(clin[tc].isna().sum()),'missing_os_event':int(clin.os_event.isna().sum()),
      'zero_times':int((time==0).sum()),'tied_event_times':int((vc>1).sum()),
      'deaths_at_tied_event_times':int(vc[vc>1].sum()),
      'stage_age_analysis_n':summary[tag]['n_samples'],
      'not_in_stage_age_analysis':len(clin)-summary[tag]['n_samples'],
      'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),
      'adjustment_source':'results/phase6_stage_age_sensitivity.json',
      'missing_age':None,'missing_stage':None,
      'missingness_note':'Pending: per-variable absence cannot be reconstructed from saved adjustment counts alone.'}
# Use deposited raw clinical sources, no API calls or new downloads.
p=ROOT/'data/raw/cptac/clinical_Pan-cancer.May2022.tsv.gz'
d=pd.read_csv(p,sep='\t').set_index('case_id')
ids=pd.read_csv(ROOT/'data/processed/cptac_luad_clinical.csv').case_id
d=d.reindex(ids)
cohort_summary['cptac'].update(missing_age=int(pd.to_numeric(d['consent/age'],errors='coerce').isna().sum()),missing_stage=int((~d['baseline/tumor_stage_pathological'].isin(['Stage I','Stage II','Stage III','Stage IV'])).sum()),missingness_note='Computed in the analyzed survival cohort from deposited CPTAC clinical data.',clinical_source=str(p.relative_to(ROOT)),clinical_source_sha256=sha(p))
# Parse GEO variables independently; preserve unusable T/N categories as missing.
import gzip,re
p=ROOT/'data/raw/geo/GSE68465_series_matrix.txt.gz'
with gzip.open(p,'rt') as f: lines=[line for line in f if line.startswith('!Sample_')]
ids=[v.strip('"') for v in next(l for l in lines if l.startswith('!Sample_geo_accession')).strip().split('\t')[1:]]
def geo_values(key):
    line=next(l for l in lines if l.startswith('!Sample_characteristics') and f'"{key}:' in l)
    return pd.Series([v.strip('"').split(': ',1)[-1] for v in line.strip().split('\t')[1:]],index=ids)
cohort_ids=pd.read_csv(ROOT/'data/processed/gse68465_luad_clinical.csv').sample_id
age=pd.to_numeric(geo_values('age').reindex(cohort_ids),errors='coerce')
stage=geo_values('disease_stage').reindex(cohort_ids)
valid=stage.fillna('').str.contains(r'pN\d') & stage.fillna('').str.contains(r'pT\d')
cohort_summary['gse68465'].update(missing_age=int(age.isna().sum()),missing_stage=int((~valid).sum()),missingness_note='Unusable pTX/pNX or absent T/N categories count as missing stage; formal M stage is unavailable.',clinical_source=str(p.relative_to(ROOT)),clinical_source_sha256=sha(p))
assert len(gse68465_stage_age().index.intersection(cohort_ids))==summary['gse68465']['n_samples']
emit('astra_r2_cohort_descriptives.json',cohort_summary)

# Historical audit: reads the pre-repair CSV, whose constant genes still carry NaN.
# The live results file has since been replaced by the repaired version.
_hist=R/'discovery_cox_naive_vs_calibrated_PRE_NANFIX.csv.bak'
saved=pd.read_csv(_hist if _hist.exists() else R/'discovery_cox_naive_vs_calibrated.csv').set_index('gene')
X,y=cohorts['cptac'];X=X.loc[saved.index]
values=X.to_numpy();sd=values.std(axis=1);constant=sd==0
z=np.divide(values-values.mean(axis=1,keepdims=True),sd[:,None],out=np.zeros_like(values),where=sd[:,None]!=0)
observed=np.array([cox_score_chi2(x,y) for x in z]); finite=saved.observed_cox_score_chi2.notna().values
assert np.array_equal(constant,~finite)
assert np.allclose(observed[finite],saved.observed_cox_score_chi2.values[finite],rtol=1e-10,atol=1e-10)
assert np.all(observed[constant]==0)
p=saved.empirical_p_value.copy();p.iloc[np.flatnonzero(constant)]=1
q=benjamini_hochberg(p.values);by=benjamini_yekutieli(p.values)
assert np.allclose(q,multipletests(p,method='fdr_bh')[1])
assert np.allclose(by,multipletests(p,method='fdr_by')[1])
corrected=saved.copy();corrected['observed_cox_score_chi2']=observed
corrected['naive_p_value']=stats.chi2.sf(observed,1)
corrected['empirical_p_value']=p;corrected['q_value_bh']=q;corrected['q_value_by']=by
corrected['survives_bh']=q<=.05;corrected['survives_by']=by<=.05
corrected.to_csv(R/'astra_r2_continuous_score_corrected.csv')
emit('astra_r2_test_audit.json',{
 'decision':'Retain continuous-covariate log-rank score and hypergeometric tie variance; not Breslow.',
 'exact_subset_score':exact,'implementation_score':actual,'breslow_score_for_diagnostic_only':bres,
 'test_checks_passed':35,
 'original_source':_hist.name if _hist.exists() else 'results/discovery_cox_naive_vs_calibrated.csv',
 'original_sha256':sha(_hist if _hist.exists() else R/'discovery_cox_naive_vs_calibrated.csv'),
 'original_constant_gene_nan_count':int(constant.sum()),'original_nan_bh_q_count':int(saved.q_value_bh.isna().sum()),
 'unchanged_finite_observed_statistics':int(finite.sum()),
 'max_observed_statistic_absolute_difference':float(np.max(np.abs(observed[finite]-saved.observed_cox_score_chi2.values[finite]))),
 'constant_genes':saved.index[constant].tolist(),
 'n_candidate_genes':len(saved),'n_nominal_hits':int((corrected.naive_p_value<.05).sum()),
 'corrected_bh_survivors':int((q<=.05).sum()),'corrected_by_survivors':int((by<=.05).sum()),
 'bh_survivor_results':corrected.loc[q<=.05,['empirical_p_value','q_value_bh','q_value_by']].to_dict(orient='index'),
 'permutations_rerun':False,'repair':'p=1/statistic=0 for verified constant genes; recalculate BH/BY over the same 11484-gene universe using saved empirical p-values for all other genes.',
 'main_logrank_results_changed':False})
# Verify preserved aggregate counts directly from saved per-gene results.
h=pd.read_csv(R/'hpa_validation_genes.csv')
mask=(h.cptac_p_naive<.05)&(h.gse68465_p_naive<.05)&(h.cptac_direction==h.hpa_direction)&(h.gse68465_direction==h.hpa_direction)
assert len(h)==1080 and mask.sum()==91 and (h.cptac_survives_bh & h.gse68465_survives_bh).sum()==2
panel=pd.read_csv(R/'table_s1_121_gene_panel.csv');assert len(panel)==121
stage=pd.read_csv(R/'phase6_stage_age_per_gene.csv')
c=stage[[f'{tag}_concordant_with_original' for tag in cohorts]].all(axis=1)
sig=(stage[[f'{tag}_adjusted_p' for tag in cohorts]]<.05).all(axis=1)
assert c.sum()==120 and (c&sig).sum()==58
print('PASS: saved HPA 91/1080 and 2 joint BH; panel 121; adjusted 120 directions and 58 joint nominal hits.',flush=True)
