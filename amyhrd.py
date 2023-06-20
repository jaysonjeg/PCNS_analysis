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
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, ttest_ind, zscore
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import acommonfuncs
from acommonvars import *
import amyhrd_utils

"""
015, 067: have very negative threshold values, so SDT unable to be estimated
031: said very high confidence for almost everyone
073: confidence ratings are mostly 0 or 100
"""


subjects_to_exclude_confidence = ['073']

to_print_subject=False
to_plot_subject=False
to_plot=True

load_table=True
robust=True #robust statistical analyses, or otherwise usual analyses (pearson r, t-test)

PT = 'sz' #patient group: 'cc', 'sz'
print(f'Analyses below compare hc with {PT}')


#outcomes,durations = get_outcomes('015',True) #015 has dramatic threshold, 073 weird confidences
#assert(0)

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

t=acommonfuncs.add_table(t,'outcomes_cface.csv')

t['group03'] = '' #make a new group column with two groups: hc and PT
for i in range(len(t)):
    if hc[i]: t.at[i,'group03'] = 'hc'
    elif eval(PT)[i]: t.at[i,'group03'] = PT


"""
r: Outer level keys are 'Intero', 'Extero'. Inner level keys are quality measures (Q_wrong_decisions_took_longer, Q_wrong_decisions_lower_confidence, Q_confidence_occurence_max, Q_HR_outlier_perc), SDT measures (dprime, criterion), psychophysics measures (threshold, slope), HR measures (bpm_mean, bpm_std) (HR not present for Extero condition)
"""

def compare(subgroup1,subgroup2,column,include_these=None,to_plot_compare=False):
    """
    Plot strip-plot of sample values in each group. Compare sample means with t-test and p-value. Also return bootstrapped confidence intervals (robust to outliers)
    """
    if include_these is None:
        include_these=t.iloc[:,0].copy()
        include_these[:]=True #array of all Trues to include all rows
    x = t.loc[t.use_hrd & eval(subgroup1) & include_these, column]
    y = t.loc[t.use_hrd & eval(subgroup2) & include_these, column]     
    mean_diff = np.mean(x)-np.mean(y)
    p_ttest = ttest_ind(x,y).pvalue
    p_MW = mannwhitneyu(x,y).pvalue
    if to_plot_compare:
        fig, ax = plt.subplots()
        sns.set_context('talk')
        sns.stripplot(ax=ax,y='group03',x=column,data=t.loc[t.use_hrd & (eval(subgroup1)|eval(subgroup2)) & include_these,:],alpha=0.5,palette=colors)
        ax.set_title(f'meandiff={mean_diff:.2f}, ttest p={p_ttest:.2f}, MW p={p_MW:.2f}')
        fig.tight_layout()
        sns.despine()
    else:
        print(f'{column}\t {subgroup1} vs {subgroup2}:\t meandiff={mean_diff:.2f}, ttest p={p_ttest:.2f}, MW p={p_MW:.2f}')

def scatter(group1,group2,column_name1,column_name2,robust=robust,include_these=None):
    """
    Scatter plot of column_name1 vs column_name2 from DataFrame t. Scatter points are colored by group1 and group2. Put correlation within each group on the title. Also plot a line of best fit for each group
    """
    if include_these is None:
        include_these=t.iloc[:,0].copy()
        include_these[:]=True #array of all Trues to include all rows
    if robust: 
        corr_func=spearmanr #could use skipped correlations in pingouin instead
        reg_func = TheilSenRegressor
    else: 
        corr_func= pearsonr
        reg_func = LinearRegression
    fig, ax = plt.subplots()
    if robust: title_string = 'Robust: '
    else: title_string = 'Non-robust: '
    for group in [group1,group2]:
        x = t.loc[t.use_hrd & include_these & eval(group),column_name1]
        y = t.loc[t.use_hrd & include_these & eval(group),column_name2]
        r,p = corr_func(x,y)
        xnew = np.linspace(min(x),max(x),100)
        reg = reg_func().fit(x.values.reshape(-1,1),y.values)
        ynew = reg.predict(xnew.reshape(-1,1))
        ax.scatter(x,y,label=group,color=colors[group])
        ax.plot(xnew,ynew,color=colors[group])
        title_string += f'{group}: r={r:.2f} p={p:.2f}, '
    ax.set_xlabel(column_name1)
    ax.set_ylabel(column_name2)
    ax.set_title(title_string[:-2])
    ax.legend([group1,group2])
    fig.tight_layout()

has_sdt = ~t.hrd_Intero_dprime.isnull()
Extero_zscores = zscore(t.hrd_Extero_threshold,nan_policy='omit')
Intero_zscores = zscore(t.hrd_Intero_threshold,nan_policy='omit')
outlier_cutoff = 3 #could use MAD-median method in pingoin instead
not_outliers = (np.abs(Extero_zscores)<outlier_cutoff) & (np.abs(Intero_zscores)<outlier_cutoff) #define outliers as being more than 3 standard deviations from the mean for exteroceptive or interoceptive thresholds
t.use_hrd = t.use_hrd & not_outliers #to exclude outliers from analyses

#Use robust regression to adjust interoceptive thresholds for exteroceptive thresholds
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
t['hrd_Intero_threshold_adj'] = np.nan
t.loc[t.use_hrd,'hrd_Intero_threshold_adj'] = t.loc[t.use_hrd,'hrd_Intero_threshold'] - reg.predict(t.loc[t.use_hrd,'hrd_Extero_threshold'].values.reshape(-1,1))

t['hrd_Intero_threshold_adj'] = t['hrd_Intero_threshold'] + 7 #a simpler way to adjust


#Absolute value of thresholds
t['hrd_Intero_threshold_abs'] = np.abs(t.hrd_Intero_threshold)
t['hrd_Extero_threshold_abs'] = np.abs(t.hrd_Extero_threshold)
t['hrd_Intero_threshold_adj_abs'] = np.abs(t.hrd_Intero_threshold_adj)


print(f'hc, n={sum(t.use_hrd & hc)}')
print(f'cc, n={sum(t.use_hrd & cc)}')
print(f'sz, n={sum(t.use_hrd & sz)}')

#df = t.loc[t.use_hrd & not_outliers & (t.group03!=''),:]
#grid=sns.pairplot(df,hue='group03',corner=False,kind='reg',x_vars=['hrd_Intero_threshold_abs','hrd_Intero_bpm_mean'],y_vars=['hrd_Extero_threshold','hrd_Extero_slope'],height=1.0,palette=colors)



#Correlations between HR measures. All highly correlated measures
grid=amyhrd_utils.pairplot(t,vars=['hrd_Intero_bpm_mean','hrd_Intero_RR_std','hrd_Intero_RMSSD','meds_chlor'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

"""
#Simple correlations between psychophysics measures. Sig: Intero_threshold corr with Extero_threshold. Distributions markedly different across groups
grid=amyhrd_utils.pairplot(t,['hrd_Intero_threshold','hrd_Intero_slope','hrd_Extero_threshold','hrd_Extero_slope'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

#Different versions of the 'threshold'. Only sig is extero_threshold_abs correlates with extero_slope
grid=amyhrd_utils.pairplot(t,x_vars=['hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Extero_threshold_abs'],y_vars=['hrd_Intero_slope','hrd_Extero_threshold','hrd_Extero_slope'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

#Do HR measures interact with Intero/Extero measures? Only hrd_Intero_threshold_adj_abs vs hrd_Intero_bpm_mean is significant
grid=amyhrd_utils.pairplot(t,x_vars = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Intero_slope'],y_vars = ['hrd_Intero_bpm_mean','hrd_Intero_RR_std'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

grid=amyhrd_utils.pairplot(t,x_vars = ['hrd_Extero_threshold','hrd_Extero_threshold_abs','hrd_Extero_slope'],y_vars = ['hrd_Intero_bpm_mean','hrd_Intero_RR_std'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

#Do clinical measures correlate with psychophyics? Sig: In HC only, high IQ correlates with high Intero_threshold. In patients only, low IQ correlates with exteroceptive imprecision and high exteroceptive |threshold|, but not with interoceptive measures
grid=amyhrd_utils.pairplot(t,x_vars = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Intero_slope'],y_vars = ['fsiq2','sofas','panss_N'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

grid=amyhrd_utils.pairplot(t,x_vars = ['hrd_Extero_threshold','hrd_Extero_threshold_abs','hrd_Extero_slope'],y_vars = ['fsiq2','sofas','panss_N'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)
"""


scatter('hc',PT,'hrd_Intero_threshold','hrd_Extero_threshold')
scatter('hc',PT,'hrd_Extero_threshold_abs','hrd_Extero_slope')

scatter('hc',PT,'hrd_Intero_bpm_mean','hrd_Intero_threshold_adj_abs')

scatter('hc',PT,'fsiq2','hrd_Extero_threshold_abs')
scatter('hc',PT,'fsiq2','hrd_Extero_slope')



#Plot intero/extero thresholds/slopes for both groups together
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


#Are psychophysics measures different across groups? Yes, for exteroceptive slope, interoceptive |threshold|, but only weak trend in interoceptive slope or exteroceptive |threshold|
for cond in ['Intero','Extero']:
    #compare('hc',PT,f'hrd_{cond}_dprime',has_sdt)
    #compare('hc',PT,f'hrd_{cond}_criterion',has_sdt)
    compare('hc',PT,f'hrd_{cond}_threshold',to_plot_compare=to_plot)
    compare('hc',PT,f'hrd_{cond}_threshold_abs')
    """
    if cond=='Intero':
        compare('hc',PT,f'hrd_{cond}_threshold_adj')
        compare('hc',PT,f'hrd_{cond}_threshold_adj_abs')
    """
    compare('hc',PT,f'hrd_{cond}_slope',to_plot_compare=to_plot)

#Do Intero/Extero and group (HC/CC) interact to determine |threshold|?. To answer this first concatenate the intero and extero thresholds abs columns vertically to a new dataframe
data = pd.DataFrame(columns=['subject','hrd_threshold_abs','hrd_threshold','hrd_slope','group','cond'])
for i in range(len(t)):
    if t.loc[i,'use_hrd'] & (hc|eval(PT))[i]:
        if hc[i]: group='hc'
        elif eval(PT)[i]: group=PT
        for cond in ['Intero','Extero']:
            dictionary = {'subject':t.at[i,'record_id'],'hrd_threshold_abs':t.at[i,f'hrd_{cond}_threshold_abs'],'hrd_threshold':t.at[i,f'hrd_{cond}_threshold'],'hrd_slope':t.at[i,f'hrd_{cond}_slope'], 'group':group,'cond':cond}
            """
            if cond=='Intero':
                dictionary = {'subject':t.at[i,'record_id'],'hrd_threshold_abs':t.at[i,f'hrd_{cond}_threshold_adj_abs'],'hrd_threshold':t.at[i,f'hrd_{cond}_threshold_adj'],'hrd_slope':t.at[i,f'hrd_{cond}_slope'], 'group':group,'cond':cond}
            """
            data=data.append(dictionary,ignore_index=True)


t['hrd_Extero_threshold_adj'] = t['hrd_Extero_threshold'] #just a copy
t['hrd_Extero_threshold_adj_abs'] = t['hrd_Extero_threshold_abs'] #just a copy


vars = ['hrd_threshold','hrd_threshold_abs','hrd_slope','hrd_threshold_adj','hrd_threshold_adj_abs']
data = pd.DataFrame(columns = list(t.columns) + vars + ['cond'])
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
            data = data.append(row,ignore_index=True)



def anova(string,columns,method='ols',return_data=False):
    data = pd.DataFrame(columns=columns)
    for i in range(len(t)):
        if t.loc[i,'use_hrd'] & (hc|eval(PT))[i]:
            if hc[i]: group='hc'
            elif eval(PT)[i]: group=PT    
            for cond in ['Intero','Extero']:
                dictionary = {'subject':t.at[i,'record_id'],'group':group,'cond':cond}
                for column in columns:
                    dictionary[column]=t.at[i,f'{column[0:3]}_{cond}_{column[4:]}']    
                data=data.append(dictionary,ignore_index=True)
    if return_data: return data
    print(string)
    if method=='ols':
        model = smf.ols(string, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table) #No significant interaction    
    elif method=='lad': #least absolute deviation (robust to outliers)
        model=smf.quantreg(string,data=data).fit(q=0.5)
        print(model.summary())

data=anova('hrd_threshold_abs ~ group + cond + group:cond', ['hrd_threshold_abs'],return_data=True)
r=pg.mixed_anova(data=data, dv='hrd_threshold_abs', within='cond', subject='subject', between='group') #sig

data=anova('hrd_threshold_adj_abs ~ group + cond + group:cond', ['hrd_threshold_adj_abs'],return_data=True)
r=pg.mixed_anova(data=data, dv='hrd_threshold_adj_abs', within='cond', subject='subject', between='group') #not sig

#Within-subjects (repeated) factors are 'cond'. Between-subject factors are group

anova('hrd_threshold_abs ~ group + cond + group:cond', ['hrd_threshold_abs']) #try with mixed effects
anova('hrd_threshold_adj_abs ~ group + cond + group:cond', ['hrd_threshold_adj_abs'])

anova('hrd_slope ~ group + cond + group:cond', ['hrd_slope'])

anova('hrd_threshold_abs ~ hrd_slope + cond + hrd_slope:cond', ['hrd_threshold_abs','hrd_slope']) 
anova('hrd_threshold_adj_abs ~ hrd_slope + cond + hrd_slope:cond', ['hrd_threshold_adj_abs','hrd_slope']) #try with mixed effects + ANCOVA

anova("hrd_threshold_adj_abs ~ hrd_slope + cond + group + hrd_slope*cond + group*cond + group*hrd_slope + group*cond*hrd_slope", ['hrd_threshold_adj_abs','hrd_slope']) #3-way



from statsmodels.stats.anova import AnovaRM
#perform the repeated measures ANOVA
print(AnovaRM(data=df, depvar='response', subject='patient', within=['drug']).fit())


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
scatter('hc',PT,'hrd_Intero_threshold_abs','cface_amp_max_mean',include_these=t.use_cface & t.cface_latencies_validperc_ha)
scatter('hc',PT,'hrd_Intero_slope','cface_amp_max_mean',include_these=t.use_cface & t.cface_latencies_validperc_ha)
scatter('hc',PT,'hrd_Intero_threshold_abs','cface_latencies_mean_ha',include_these=t.use_cface & t.cface_latencies_validperc_ha)
scatter('hc',PT,'hrd_Intero_slope','cface_latencies_mean_ha',include_these=t.use_cface & t.cface_latencies_validperc_ha)
amyhrd_utils.pairplot(t,['hrd_Intero_threshold_abs','hrd_Extero_threshold_abs','hrd_Intero_slope','cface_amp_max_mean','cface_latencies_mean_ha','cface_latencies_mean_an'],include_these=t.use_hrd & t.use_cface & not_outliers)
"""

plt.show(block=False)