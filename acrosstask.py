"""
From each task, we have obtained some outcome measures for each subject, for example the mean facial expressiveness during movie viewing. This script compares outcome measures from different tasks and modalities
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from acommonvars import *
import acommonfuncs


#Important outcomes: age_years, panss_P, panss_N, panss_bluntedaffected, meds_chlor, cgi_s, sofas, fsiq2, sas

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

if False:
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


group='group03'
t = t[t[group].values!=''] #remove subjects with no group
gps=list(t[group].unique()) 


def compare(df,group,outcomes,dim_subplots):
    """
    df is a dataframe. group is the column name for grouping variable. outcomes is a list of column names in df. dim_subplots is a tuple of the dimensions of the subplot grid. For each outcome, strip-plot of distribution for each clinical group, and t-test of difference.
    Make sure that there are no empty strings '' in the group column
    """
    fig,axs = plt.subplots(*dim_subplots,figsize=(16,8))
    for i in range(len(outcomes)):
        string=outcomes[i]
        ax = axs[np.unravel_index(i,dim_subplots)]
        sns.stripplot(ax=ax,data=df,x=group,hue=group,palette=colors,y=string)
        group1 = df.loc[df[group]==gps[0],string].dropna()
        group2 = df.loc[df[group]==gps[1],string].dropna()
        diff = group1.mean() - group2.mean()
        p_ttest = stats.ttest_ind(group1,group2).pvalue
        ax.set_title(f'diff {diff:.2f} p={p_ttest:.2f}')
        ax.set_ylabel(string)
    fig.tight_layout()


compare(t.loc[t.use_prop & ~t.prop_outliers,:], group, ['prop_stim_max','prop_slope','prop_r2'], (2,2))

compare(t.loc[t.use_sinus & ~t.sinus_outliers,:], group, ['sinus_amp_initial','sinus_plv_initial','sinus_lag_hilbert_initial_abs','sinus_lag_hilbert_initial', 'sinus_rapidity_initial'], (2,3))

compare(t.loc[t.use_cface,:],group, ['cface_amp_max_mean', 'cface_latencies_mean_ha', 'cface_latencies_mean_an','cface_maxgrads_mean_ha','cface_maxgrads_mean_an'],(2,3))

compare(t.loc[t.use_hrd & ~t.hrd_outliers],group, ['hrd_Intero_threshold','hrd_Extero_threshold','hrd_Intero_threshold_abs','hrd_Intero_slope','hrd_Extero_slope','hrd_Intero_meta_d','hrd_Intero_m_ratio','hrd_Intero_bpm_mean','hrd_Intero_RR_std'],(3,3))

compare(t,group,['age_years', 'fsiq2'],(2,2))

plt.show(block=False)