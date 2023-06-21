"""
Anayse myHRD
Basically Copied myJupHeartRateDiscrimination.ipynb
Needs conda environment 'hr'
PPG was actually recorded at 100Hz but resampled to 1000Hz before saving as signal.txt. Signal.txt only contains PPG signals during HR listening time for interoceptive condition (5 sec per trial, but somehow saved 6sec). So in total PPG represents 6sec * 40 trials = 240 sec of HR recording. PPG in signal.txt is not continguous.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore
from sklearn.linear_model import TheilSenRegressor
import acommonfuncs
from acommonvars import *
import amyhrd_utils

"""
015, 067: have very negative threshold values, so SDT unable to be estimated
031: said very high confidence for almost everyone
073: confidence ratings are mostly 0 or 100
"""


subjects_to_exclude_confidence = ['073']

to_print_subject=True
to_plot_subject=True
to_plot=True

load_table=True
robust=True #robust statistical analyses, or otherwise usual analyses (pearson r, t-test)

PT = 'cc' #patient group: 'cc', 'sz'
print(f'Analyses below compare hc with {PT}')

intero_adjust_method = 'add7' #'regress_extero' or 'add7'
print(f'Adjusting interoceptive thresholds by {intero_adjust_method}')

outlier_method = 'zscore' #'zscore' or 'madmedianrule'
outlier_cutoff = 3 #z-score for outlier cutoff. Could use MAD-median method in pingoin instead


from glob import glob
from metadPy import sdt
from metadPy.utils import trials2counts, discreteRatings
from metadPy.plotting import plot_confidence, plot_roc
subject='065'
r={'Intero':{},'Extero':{}} #stores outcome measures for this subject
contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\HRD*")
assert(len(contents)==1)
resultsFolder=contents[0]
df=pd.read_csv(glob(f"{resultsFolder}\\*final.txt")[0]) # Logs dataframe
interoPost=np.load(glob(f"{resultsFolder}\\*Intero_posterior.npy")[0]) # History of posteriors distribution
exteroPost=np.load(glob(f"{resultsFolder}\\*Extero_posterior.npy")[0])
signal_df=pd.read_csv(glob(f"{resultsFolder}\\*signal.txt")[0]) # PPG signal
signal_df['Time'] = np.arange(0, len(signal_df))/1000 # Create time vector <--- assumes 1000Hz sampling rate which is wrong. We used 100 Hz
df_old=df
df = df[df.RatingProvided == 1] #removing trials where no rating was provided
this_df = df[df.Modality=='Intero']
new_confidence, _ = discreteRatings(this_df.Confidence)  


for i, cond in enumerate(['Intero', 'Extero']):
    this_df = df[df.Modality == cond].copy()
    this_df['Stimuli'] = (this_df.responseBPM > this_df.listenBPM)
    this_df['Responses'] = (this_df.Decision == 'More')
    hits, misses, fas, crs = sdt.scores(data=this_df)
    if hits==0 or crs==0:
        d, c = np.nan, np.nan
    else:
        hr, far = sdt.rates(data=this_df,hits=hits, misses=misses, fas=fas, crs=crs)
        d, c = sdt.dprime(data=this_df,hit_rate=hr, fa_rate=far), sdt.criterion(data=this_df,hit_rate=hr, fa_rate=far)
        if to_print_subject:
            print(f'Condition: {cond} - d-prime: {d:.2f} - criterion: {c:.2f}')
    r[cond]['dprime']=d
    r[cond]['criterion']=c


sns.set_context('talk')
fig, axs = plt.subplots(2, 2, figsize=(13, 5))
for i, cond in enumerate(['Intero', 'Extero']):
    this_df = df[df.Modality == cond]
    this_df = this_df[~this_df.Confidence.isnull()]

    new_confidence, _ = discreteRatings(this_df.Confidence) # discretize confidence ratings into 4 bins
    this_df['Confidence'] = new_confidence
    this_df['Stimuli'] = (this_df.Alpha > 0).astype('int')
    this_df['Responses'] = (this_df.Decision == 'More').astype('int')
    nR_S1, nR_S2 = trials2counts(data=this_df)
    plot_confidence(nR_S1, nR_S2, ax=axs[0,i])
    axs[0,i].set_title(f'{cond} condition')


    from metadPy.plotting import plot_roc
    plot_roc(nR_S1, nR_S2, ax=axs[1,i])
    print(f'roc auc for {cond} is {this_df.roc_auc(nRatings=4):.2f}')

    from metadPy.mle import metad
    this_df['Accuracy']=(this_df['Stimuli']==this_df['Responses']).astype(int) 
    z=metad(data=this_df,nRatings=4,stimuli='Stimuli',accuracy='Accuracy',confidence='Confidence',verbose=0) #estimate meta dprime using MLE

    from metadPy.bayesian import hmetad
    model, trace = hmetad(
    data=this_df, nRatings=4, stimuli='Stimuli',
    accuracy='Accuracy', confidence='Confidence'
    )

plt.show(block=False)
assert(0)


outcomes,durations = amyhrd_utils.get_outcomes('015',to_print_subject=True,to_plot_subject=False) #015 has dramatic threshold, 073 weird confidences


t['use_hrd'] = ((include) & (t.valid_hrdo==1)) #those subjects whose HRD data we will use
if load_table:
    t = pd.read_csv(f'{temp_folder}\\outcomes_myhrd.csv')
else:
    for i in range(t.shape[0]): 
        if t.use_hrd[i]:
            print(subs[i])
            outcomes, task_duration = amyhrd_utils.get_outcomes(subs[i],to_print_subject,to_plot_subject)
            for cond in ['Intero','Extero']:
                for j in outcomes[cond].keys():
                    t.loc[i,f'hrd_{cond}_{j}']=outcomes[cond][j]
    t.to_csv(f'{temp_folder}\\outcomes_myhrd.csv')

#t=acommonfuncs.add_table(t,'outcomes_cface.csv')

t['group03'] = '' #make a new group column with two groups: hc and PT
for i in range(len(t)):
    if hc[i]: t.at[i,'group03'] = 'hc'
    elif eval(PT)[i]: t.at[i,'group03'] = PT

"""
r: Outer level keys are 'Intero', 'Extero'. Inner level keys are quality measures (Q_wrong_decisions_took_longer, Q_wrong_decisions_lower_confidence, Q_confidence_occurence_max, Q_HR_outlier_perc), SDT measures (dprime, criterion), psychophysics measures (threshold, slope), HR measures (bpm_mean, RR_std,RMSSD) (HR not present for Extero condition)
"""

has_sdt = ~t.hrd_Intero_dprime.isnull()


if outlier_method=='zscore':
    Extero_zscores = zscore(t.hrd_Extero_threshold,nan_policy='omit')
    Intero_zscores = zscore(t.hrd_Intero_threshold,nan_policy='omit')
    HR_zscores = zscore(t.hrd_Intero_bpm_mean,nan_policy='omit')
    not_outliers = (np.abs(Extero_zscores)<outlier_cutoff) & (np.abs(Intero_zscores)<outlier_cutoff) #define outliers as being more than 3 standard deviations from the mean for exteroceptive or interoceptive thresholds
elif outlier_method=='madmedianrule':
    Extero_out=pg.madmedianrule(t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_threshold']) 
    Intero_out=pg.madmedianrule(t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_threshold']) 
    HR_out=pg.madmedianrule(t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_bpm_mean']) 
    madmedian_outliers = Extero_out | Intero_out | HR_out
    not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & (hc|eval(PT)),'record_id'],madmedian_outliers)
t.use_hrd = t.use_hrd & not_outliers #to exclude outliers from analyses

print(f'hc, n={sum(t.use_hrd & hc)}')
print(f'cc, n={sum(t.use_hrd & cc)}')
print(f'sz, n={sum(t.use_hrd & sz)}')
print(f'PT, n={sum(t.use_hrd & eval(PT))}')

#Metacognition stuff

#Print quality checks
for cond in ['Intero','Extero']:
    for measure in ['wrong_decisions_took_longer','wrong_decisions_lower_alpha_pos','wrong_decisions_lower_alpha_neg','wrong_decisions_lower_confidence','confidence_occurence_max']:
        string = f'hrd_{cond}_Q_{measure}'
        values = t.loc[t.use_hrd,string]
        print(f'{cond} {measure} \t {np.mean(values):.2f}')
values = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_Q_HR_outlier_perc']
print(f'Intero HR outlier perc \t {np.mean(values):.2f}')

#Get confidence rating outliers
Intero_confidence_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_Q_confidence_occurence_max'] > 0.65
Extero_confidence_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_confidence_occurence_max'] > 0.65
confidence_out = Intero_confidence_out | Extero_confidence_out
confidence_not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & (hc|eval(PT)),'record_id'],confidence_out)

#Get other quality check measures
Q1_Extero_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_wrong_decisions_took_longer'] == False
Q2_Extero_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_wrong_decisions_lower_alpha_pos'] == False
Q3_Extero_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_wrong_decisions_lower_alpha_neg'] == False
Q_Extero_out = Q1_Extero_out | Q2_Extero_out | Q3_Extero_out
Q_Extero_not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & (hc|eval(PT)),'record_id'],Q_Extero_out)






#Use robust regression to predict interoceptive thresholds from exteroceptive thresholds
x = t.loc[t.use_hrd & not_outliers,'hrd_Extero_threshold']
y = t.loc[t.use_hrd & not_outliers,'hrd_Intero_threshold']
xnew = np.linspace(min(x),max(x),100)
reg = TheilSenRegressor().fit(x.values.reshape(-1,1),y.values)
ynew = reg.predict(xnew.reshape(-1,1))
fig,ax=plt.subplots()
ax.scatter(x,y)
ax.plot(xnew,ynew)
ax.set_xlabel('Exteroceptive threshold')
ax.set_ylabel('Interoceptive threshold')
ax.set_title(f'Theil-Sen regression: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}')

if intero_adjust_method=='regress_extero':
    t['hrd_Intero_threshold_adj'] = np.nan
    t.loc[t.use_hrd,'hrd_Intero_threshold_adj'] = t.loc[t.use_hrd,'hrd_Intero_threshold'] - reg.predict(t.loc[t.use_hrd,'hrd_Extero_threshold'].values.reshape(-1,1))
elif intero_adjust_method=='add7':
    t['hrd_Intero_threshold_adj'] = t['hrd_Intero_threshold'] + 7 #a simpler way to adjust

#Absolute value of thresholds
t['hrd_Intero_threshold_abs'] = np.abs(t.hrd_Intero_threshold)
t['hrd_Extero_threshold_abs'] = np.abs(t.hrd_Extero_threshold)
t['hrd_Intero_threshold_adj_abs'] = np.abs(t.hrd_Intero_threshold_adj)

#Correlations between HR measures. All highly correlated measures
grid=acommonfuncs.pairplot(t,vars=['hrd_Intero_bpm_mean','hrd_Intero_RR_std','hrd_Intero_RMSSD','meds_chlor'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

"""
#Simple correlations between psychophysics measures. Sig: Intero_threshold corr with Extero_threshold. Distributions markedly different across groups
grid=acommonfuncs.pairplot(t,['hrd_Intero_threshold','hrd_Intero_slope','hrd_Extero_threshold','hrd_Extero_slope'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

#Different versions of the 'threshold'. Only sig is extero_threshold_abs correlates with extero_slope
grid=acommonfuncs.pairplot(t,x_vars=['hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Extero_threshold_abs'],y_vars=['hrd_Intero_slope','hrd_Extero_threshold','hrd_Extero_slope'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

#Do HR measures interact with Intero/Extero measures? Only hrd_Intero_threshold_adj_abs vs hrd_Intero_bpm_mean is significant
grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Intero_slope'],y_vars = ['hrd_Intero_bpm_mean','hrd_Intero_RR_std'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Extero_threshold','hrd_Extero_threshold_abs','hrd_Extero_slope'],y_vars = ['hrd_Intero_bpm_mean','hrd_Intero_RR_std'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

#Do clinical measures correlate with psychophyics? Sig: In HC only, high IQ correlates with high Intero_threshold. In patients only, low IQ correlates with exteroceptive imprecision and high exteroceptive |threshold|, but not with interoceptive measures
grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Intero_slope'],y_vars = ['fsiq2','sofas','panss_N'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Extero_threshold','hrd_Extero_threshold_abs','hrd_Extero_slope'],y_vars = ['fsiq2','sofas','panss_N'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)
"""

#Scatterplots of 2 variables, for controls and patient groups
amyhrd_utils.scatter(t,'hc',PT,'hrd_Intero_threshold','hrd_Extero_threshold',robust=robust)
amyhrd_utils.scatter(t,'hc',PT,'hrd_Extero_threshold_abs','hrd_Extero_slope',robust=robust)

amyhrd_utils.scatter(t,'hc',PT,'hrd_Intero_bpm_mean','hrd_Intero_threshold_adj_abs',robust=robust)

amyhrd_utils.scatter(t,'hc',PT,'fsiq2','hrd_Extero_threshold_abs',robust=robust)
amyhrd_utils.scatter(t,'hc',PT,'fsiq2','hrd_Extero_slope',robust=robust)


#Are psychophysics measures different across groups? 
for cond in ['Intero','Extero']:
    #compare('hc',PT,f'hrd_{cond}_dprime',has_sdt)
    #compare('hc',PT,f'hrd_{cond}_criterion',has_sdt)
    amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold',to_plot_compare=to_plot)
    amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold_abs')  
    if cond=='Intero':
        amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold_adj')
        amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold_adj_abs')
    amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_slope',to_plot_compare=to_plot)

#Plot intero/extero thresholds/slopes for both groups together
"""
fig,ax=plt.subplots()
for group in ['hc',PT]:
    for cond in ['Intero','Extero']:
        x = t.loc[t.use_hrd & eval(group),f'hrd_{cond}_slope']
        y = t.loc[t.use_hrd & eval(group),f'hrd_{cond}_threshold']
        if cond == 'Intero':
            y = t.loc[t.use_hrd & eval(group),f'hrd_{cond}_threshold_adj']
        ax.scatter(x,y,label=f'{group} {cond}',color=colors[group],marker={'Intero':'.','Extero':'x'}[cond])
ax.set_xlabel('slope')
ax.set_ylabel('threshold')
ax.legend()
"""

if intero_adjust_method=='regress_extero':
    #Effect disappears in linear model when including HR
    print(sm.stats.anova_lm(smf.ols('hrd_Intero_threshold_adj_abs ~ group03 + hrd_Intero_bpm_mean', data=t.loc[t.use_hrd & (hc|eval(PT)),:]).fit(), typ=2))

    #But HR doesn't account for interoceptive thresholds in controls
    amyhrd_utils.scatter(t,'hc',PT,'hrd_Intero_bpm_mean','hrd_Intero_threshold_adj_abs',robust=robust)
    amyhrd_utils.scatter(t,'hc',PT,'hrd_Intero_bpm_mean','hrd_Intero_threshold_adj',robust=robust)

elif intero_adjust_method == 'add7':

    #Make new table t2, where interoceptive and exteroceptive thresholds are in the same column, and there is a new column 'cond' which is either 'Intero' or 'Extero'
    t['hrd_Extero_threshold_adj'] = t['hrd_Extero_threshold'] #just a copy
    t['hrd_Extero_threshold_adj_abs'] = t['hrd_Extero_threshold_abs'] #just a copy
    vars = ['hrd_threshold','hrd_threshold_abs','hrd_slope','hrd_threshold_adj','hrd_threshold_adj_abs']
    t2 = pd.DataFrame(columns = list(t.columns) + vars + ['cond'])
    for i in range(len(t)):
        if t.loc[i,'use_hrd'] & (hc|eval(PT))[i]:
            if hc[i]: group='hc'
            elif eval(PT)[i]: group=PT    
            for cond in ['Intero','Extero']:
                row = dict(t.loc[i,:])
                #row.pop('Unnamed: 0')
                #row['group'] = group
                row['cond'] = cond
                for var in vars:
                    old_column_name = f'{var[0:3]}_{cond}_{var[4:]}'
                    row[var] = t.at[i,old_column_name]
                t2 = t2.append(row,ignore_index=True)

    #pingoin's mixed_anova
    print(pg.mixed_anova(data=t2, dv='hrd_threshold_abs', within='cond', subject='record_id', between='group03')) #sig
    print(pg.mixed_anova(data=t2, dv='hrd_threshold_adj_abs', within='cond', subject='record_id', between='group03',correction=True)) #not sig

    #As above but ANOVAs

    print(sm.stats.anova_lm(smf.ols('hrd_Intero_threshold_adj_abs ~ group03 + hrd_Intero_bpm_mean', data=t.loc[t.use_hrd & (hc|eval(PT)),:]).fit(), typ=2))

    print(sm.stats.anova_lm(smf.ols('hrd_threshold_adj_abs ~ group03 + cond + group03:cond', data=t2).fit(), typ=2)) #sig with sz, adj
    print(sm.stats.anova_lm(smf.ols('hrd_threshold_adj_abs ~ hrd_slope + cond + hrd_slope:cond', data=t2).fit(), typ=2)) 
    print(sm.stats.anova_lm(smf.ols("hrd_threshold_adj_abs ~ hrd_slope + cond + group03 + hrd_slope*cond + group03*cond + group03*hrd_slope + group03*cond*hrd_slope", data=t2).fit(), typ=2)) #3-way. sig with adj
    print(sm.stats.anova_lm(smf.ols('hrd_slope ~ group03 + cond + group03:cond', data=t2).fit(), typ=2))

    #As above, but using random intercepts for subject
    # hrd_threshold_adj_abs ~ cond + group + group*cond + (1|record_id)
    mdf=smf.mixedlm("hrd_threshold_adj_abs ~ cond+group03+cond*group03", t2, groups=t2["record_id"]).fit().summary()
    mdf=smf.mixedlm("hrd_threshold_adj_abs ~ cond+group03+cond*group03+hrd_Intero_bpm_mean", t2, groups=t2["record_id"],re_formula="~cond").fit().summary()

    mdf=smf.mixedlm('hrd_threshold_adj_abs ~ hrd_slope+cond+hrd_slope*cond',t2,groups=t2['record_id']).fit().summary()
    mdf=smf.mixedlm("hrd_threshold_adj_abs ~ cond+group03+hrd_slope+cond*group03+cond*hrd_slope+group03*hrd_slope+cond*group03*hrd_slope", t2, groups=t2["record_id"]).fit().summary()
    mdf=smf.mixedlm('hrd_slope ~ group03+cond+group03:cond',t2,groups=t2['record_id']).fit().summary()


#Is it the same subjects having high intero and extero thresholds? Plot Intero and Extero thresholds for each subject with a line connecting them
"""
fig,ax=plt.subplots()
pointplot_kwargs={'marker':'.','scale':0.2}
pg.plot_paired(ax=ax,data=data.loc[data.group=='hc',:],dv='hrd_threshold',within='cond',subject='subject',boxplot=False,boxplot_in_front=True,colors=[colors['hc']]*3,pointplot_kwargs=pointplot_kwargs)
pg.plot_paired(ax=ax,data=data.loc[data.group==PT,:],dv='hrd_threshold',within='cond',subject='subject',boxplot=False,boxplot_in_front=True,colors=[colors[PT]]*3,pointplot_kwargs=pointplot_kwargs)
ax.legend(['hc',PT],labelcolor=[colors['hc'],colors[PT]])
fig.tight_layout()
sns.despine()
"""


"""
#Correlations between psychophysics measures and cface task measures
scatter('hc',PT,'hrd_Intero_threshold_abs','cface_amp_max_mean',include_these=t.use_cface & t.cface_latencies_validperc_ha)
scatter('hc',PT,'hrd_Intero_slope','cface_amp_max_mean',include_these=t.use_cface & t.cface_latencies_validperc_ha)
scatter('hc',PT,'hrd_Intero_threshold_abs','cface_latencies_mean_ha',include_these=t.use_cface & t.cface_latencies_validperc_ha)
scatter('hc',PT,'hrd_Intero_slope','cface_latencies_mean_ha',include_these=t.use_cface & t.cface_latencies_validperc_ha)
acommonfuncs.pairplot(t,['hrd_Intero_threshold_abs','hrd_Extero_threshold_abs','hrd_Intero_slope','cface_amp_max_mean','cface_latencies_mean_ha','cface_latencies_mean_an'],include_these=t.use_hrd & t.use_cface & not_outliers)
"""

plt.show(block=False)