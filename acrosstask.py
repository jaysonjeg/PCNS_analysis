"""
From each task, we have obtained some outcome measures for each subject, for example the mean facial expressiveness during movie viewing. This script compares outcome measures from different tasks and modalities
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from acommonvars import *
import acommonfuncs


#Important outcomes: age_years, panss_P, panss_N, panss_bluntedaffected, meds_chlor, cgi_s, sofas, fsiq2, sas


if True:
    #facegng task: gng_RT_calm, gng_RT_fear
    acommonfuncs.add_table(t,'outcomes_facegng.csv')

if True:
    #Task prop: prop_stim_max, prop_slope, prop_r2
    cols_prop = ['use_prop','prop_outliers','prop_stim', 'prop_resp','prop_stim_min','prop_stim_max','prop_stim_range','prop_slope','prop_intercept','prop_r2']
    acommonfuncs.add_table(t,'outcomes_prop.csv')
    acommonfuncs.str_columns_to_literals(t,['prop_stim', 'prop_resp'])

if True:
    #Task sinus: sinus_amp_initial, sinus_plv_initial, sinus_lag_hilbert_initial (abs)
    cols_sinus = ['use_sinus','sinus_outliers']
    for string in ['blinks', 'amp', 'rapidity', 'lag_hilbert', 'lag_fourier']:
        cols_sinus.append(f'sinus_{string}_initial')
        cols_sinus.append(f'sinus_{string}_final')
    for string in ['corrs', 'plv']:
        cols_sinus.append(f'sinus_{string}_initial')
        cols_sinus.append(f'sinus_{string}_jump')
        cols_sinus.append(f'sinus_{string}_final')
    acommonfuncs.add_table(t,'outcomes_sinus.csv')
    t['sinus_lag_hilbert_initial_abs'] = np.abs(t.sinus_lag_hilbert_initial)

t = t.copy()

if True:
    #Task movie Ricky: ricky_sr_AU06_mean, ricky_sr_AU12_mean, ricky_dc_AU17_mean, ricky_dc_AU23_mean
    cols_ricky = ['use_ricky','ricky_outliers']
    for nAU in range(len(aus_labels)):
        action_unit = aus_labels[nAU]
        cols_ricky.append(f'ricky_sr_{action_unit}_mean')
        cols_ricky.append(f'ricky_sr_{action_unit}_cv')
    cols_ricky.append(f'ricky_dc_{action_unit}_mean')
    acommonfuncs.add_table(t,'outcomes_ricky.csv')

if True:
    #Task cface face data: cface_amp_max_mean, cface_latencies_mean_ha, cface_latencies_mean_an
    #no outliers
    acommonfuncs.add_table(t,'outcomes_cface_face.csv')
    acommonfuncs.str_columns_to_literals(t,['cface_mean_ts_ha_pca0','cface_mean_ts_an_pca0','cface_mean_ts_ha_au12'])

if True:
    #HRD task: hrd_Intero_bpm_mean, hrd_Intero_RR_std, hrd_Intero_threshold_abs, hrd_Intero_slope
    acommonfuncs.add_table(t,'outcomes_myhrd_reduced.csv')


t=t.copy()

robust=False
group='group02'
t = t[t[group].values!=''] #remove subjects with no group
gps=list(t[group].unique()) 

if False:
    acommonfuncs.compare_multi(t.loc[t.use_facegng,:], group, ['gng_RT_calm','gng_RT_fear'], (2,2))
    acommonfuncs.compare_multi(t.loc[t.use_prop & ~t.prop_outliers,:], group, ['prop_stim_max','prop_slope','prop_r2'], (2,2))
    acommonfuncs.compare_multi(t.loc[t.use_sinus & ~t.sinus_outliers,:], group, ['sinus_amp_initial','sinus_plv_initial','sinus_lag_hilbert_initial_abs','sinus_lag_hilbert_initial', 'sinus_rapidity_initial'], (2,3))
    acommonfuncs.compare_multi(t.loc[t.use_cface,:],group, ['cface_amp_max_mean', 'cface_latencies_mean_ha', 'cface_latencies_mean_an','cface_maxgrads_mean_ha','cface_maxgrads_mean_an'],(2,3))
    acommonfuncs.compare_multi(t.loc[t.use_hrd & ~t.hrd_outliers],group, ['hrd_Intero_threshold','hrd_Extero_threshold','hrd_Intero_threshold_abs','hrd_Intero_slope','hrd_Extero_slope','hrd_Intero_meta_d','hrd_Intero_m_ratio','hrd_Intero_bpm_mean','hrd_Intero_RR_std'],(3,3))
    acommonfuncs.compare_multi(t,group,['age_years', 'fsiq2'],(2,2))



facegng_outcomes = ['gng_RT_calm','gng_RT_fear']
prop_outcomes = ['prop_stim_max','prop_slope','prop_r2']
sinus_outcomes = ['sinus_amp_initial','sinus_plv_initial','sinus_lag_hilbert_initial','sinus_lag_hilbert_initial_abs', 'sinus_rapidity_initial']
ricky_outcomes = ['ricky_sr_AU06_mean', 'ricky_sr_AU12_mean','ricky_dc_AU17_mean', 'ricky_dc_AU23_mean']
cface_outcomes = ['cface_amp_max_mean', 'cface_latencies_mean_ha', 'cface_latencies_mean_an']
hrd_outcomes = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_bpm_mean','hrd_Intero_RR_std']

demo_outcomes = ['age_years', 'fsiq2']
clinical_outcomes = ['panss_P', 'panss_N', 'panss_bluntedaffect', 'meds_chlor', 'cgi_s', 'sofas']

from itertools import product

if False:
    pairs = list(product(demo_outcomes,facegng_outcomes))
    t2=t.loc[t.use_facegng,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,2),figsize=(10,9))

    pairs = list(product(demo_outcomes,prop_outcomes))
    t2=t.loc[t.use_prop & ~t.prop_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,3),figsize=(10,9))

    pairs = list(product(demo_outcomes,sinus_outcomes))
    t2=t.loc[t.use_sinus & ~t.sinus_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,5),figsize=(10,9))

    pairs = list(product(demo_outcomes,ricky_outcomes))
    t2=t.loc[t.use_ricky & ~t.ricky_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,4),figsize=(10,9))

    pairs = list(product(demo_outcomes,cface_outcomes))
    t2=t.loc[t.use_cface,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,3),figsize=(10,9))

    pairs = list(product(demo_outcomes,hrd_outcomes))
    t2=t.loc[t.use_hrd & ~t.hrd_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,4),figsize=(10,9))

if True:
    pairs = list(product(clinical_outcomes,facegng_outcomes))
    t2=t.loc[(t[group]!='hc') & t.use_facegng,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(6,2),figsize=(10,9))

    pairs = list(product(clinical_outcomes,prop_outcomes))
    t2=t.loc[(t[group]!='hc') & t.use_prop & ~t.prop_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(6,3),figsize=(10,9))

    pairs = list(product(clinical_outcomes,sinus_outcomes))
    t2=t.loc[(t[group]!='hc') & t.use_sinus & ~t.sinus_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(6,5),figsize=(10,9))

    pairs = list(product(clinical_outcomes,ricky_outcomes))
    t2=t.loc[(t[group]!='hc') & t.use_ricky & ~t.ricky_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(6,4),figsize=(10,9))

    pairs = list(product(clinical_outcomes,cface_outcomes))
    t2=t.loc[(t[group]!='hc') & t.use_cface,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(6,3),figsize=(10,9))

    pairs = list(product(clinical_outcomes,hrd_outcomes))
    t2=t.loc[(t[group]!='hc') & t.use_hrd & ~t.hrd_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(6,4),figsize=(10,9))

if False:
    pairs = list(product(facegng_outcomes,prop_outcomes))
    t2=t.loc[t.use_facegng & t.use_prop & ~t.prop_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,3))

    pairs = list(product(facegng_outcomes,sinus_outcomes))
    t2=t.loc[t.use_facegng & t.use_sinus & ~t.sinus_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,5))

    pairs = list(product(facegng_outcomes,ricky_outcomes))
    t2=t.loc[t.use_facegng & t.use_ricky & ~t.ricky_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,4))

    pairs = list(product(facegng_outcomes,cface_outcomes))
    t2=t.loc[t.use_facegng & t.use_cface,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,3))

    pairs = list(product(facegng_outcomes,hrd_outcomes))
    t2=t.loc[t.use_facegng & t.use_hrd & ~t.hrd_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(2,4))

if False: 
    pairs = list(product(prop_outcomes,sinus_outcomes))
    t2=t.loc[t.use_sinus & ~t.sinus_outliers & t.use_prop & ~t.prop_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(3,5))

    pairs = list(product(prop_outcomes,ricky_outcomes))
    t2=t.loc[t.use_ricky & ~t.ricky_outliers & t.use_prop & ~t.prop_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(3,4))

    pairs = list(product(prop_outcomes,cface_outcomes))
    t2=t.loc[t.use_prop & ~t.prop_outliers & t.use_cface,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(3,3))

    pairs = list(product(prop_outcomes,hrd_outcomes))
    t2=t.loc[t.use_prop & ~t.prop_outliers & t.use_hrd & ~t.hrd_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(3,4))

if False:
    pairs = list(product(sinus_outcomes,ricky_outcomes))
    t2=t.loc[t.use_sinus & ~t.sinus_outliers & t.use_ricky & ~t.ricky_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(5,4))

    pairs = list(product(sinus_outcomes,cface_outcomes))
    t2=t.loc[t.use_sinus & ~t.sinus_outliers & t.use_cface,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(5,3))

    pairs = list(product(sinus_outcomes,hrd_outcomes))
    t2=t.loc[t.use_sinus & ~t.sinus_outliers & t.use_hrd & ~t.hrd_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(5,4))

if False:
    pairs = list(product(ricky_outcomes,cface_outcomes))
    t2=t.loc[t.use_ricky & ~t.ricky_outliers & t.use_cface,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(4,3))

    pairs = list(product(ricky_outcomes,hrd_outcomes))
    t2=t.loc[t.use_ricky & ~t.ricky_outliers & t.use_hrd & ~t.hrd_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(4,4))

    pairs = list(product(cface_outcomes,hrd_outcomes))
    t2=t.loc[t.use_cface & t.use_hrd & ~t.hrd_outliers,:]
    acommonfuncs.scatter_multi(t2,group,pairs,robust=robust,dim=(5,4))

plt.show(block=False)