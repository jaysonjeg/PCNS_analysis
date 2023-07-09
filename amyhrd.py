"""
Anayse myHRD
Basically Copied myJupHeartRateDiscrimination.ipynb
Needs conda environment 'hr'
PPG was actually recorded at 100Hz but resampled to 1000Hz before saving as signal.txt. Signal.txt only contains PPG signals during HR listening time for interoceptive condition (5 sec per trial, but somehow saved 6sec). So in total PPG represents 6sec * 40 trials = 240 sec of HR recording. PPG in signal.txt is not continguous.

Adds new columns to t, prefixed by given by "hrd_{Modality}_{outcome}
Modality can be "Intero" or "Extero"
Outcomes:
    Psychometric function values
        Final estimates using psi staircase: threshold_psi, slope_psi
        Fitted, single subject, Bayesian: threshold_Bay, slope_Bay (exclude HR outliers)
    SDT outcomes: dprime, criterion
    Metacognition (all exclude HR outliers)
        Single subject MLE: meta_d_MLE, m_ratio_MLE
        Single subject Bayesian: dprime_Bay, meta_d_Bay, m_ratio_Bay
        Group level MLE (NOT WORKING): meta_d_groupMLE, m_ratio_groupMLE
    HR outcomes: (only with "Intero" modality)
        bpm_mean
        RR_std
        RMSSD

Quaity control outcomes begin with hrd_{Modality}_Q_{outcome}
    wrong_decisions_lower_alpha_pos
    wrong_decisions_lower_alpha_neg
    wrong_decisions_lower_confidence
    confidence_occurence_max: The maximum proportional occurence of any of the 4 confidence categories (out of 1.0)


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
073: confidence ratings are mostly 0 or 100. Make sure this is excluded
"""

if __name__=='__main__':
    clock = acommonfuncs.clock()

    #SETTABLE PARAMETERS
    to_print_subject=False
    to_plot_subject=False
    to_plot=True

    load_df=True
    load_t=True
    robust=True #robust statistical analyses, or otherwise usual analyses (pearson r, t-test)

    PT = 'cc' #patient group: 'cc', 'sz'
    print(f'Analyses below compare hc with {PT}')

    psychometric_type = 'Bay' #psi or Bay
    metacog_type = 'Bay' #MLE or Bay

    intero_adjust_method = 'none' #'regress_extero', 'add7', 'none'
    print(f'Adjusting interoceptive thresholds by {intero_adjust_method}')

    outlier_method = 'zscore' #'zscore' or 'madmedianrule'
    outlier_cutoff = 3 #z-score for outlier cutoff. Could use MAD-median method in pingoin instead
    print(f'Outlier rule is {outlier_method}')

    #t=acommonfuncs.add_table(t,'outcomes_cface.csv')

    #outcomes,durations, df = amyhrd_utils.get_outcomes('130',to_print_subject=False,to_plot_subject=False) #015 has dramatic threshold, 073 weird confidences
    #print(f'Done at {clock.time()[1]}')
    #assert(0)

    """
    Get data tables, returning the following
    df: each row is a single trial of a subject. Subjects are concatenated vertically. Contains trial-wise measures
    t: each row is the entire data for a single subject. Contains summary measures.
    """

    t['use_hrd'] = ((include) & (t.valid_hrdo==1)) #those subjects whose HRD data we will use
    if load_df:
        df = pd.read_csv(f'{temp_folder}\\outcomes_myhrd_df.csv')
    else:
        list_of_df = [amyhrd_utils.get_trialwise_outcomes(subs[i])[0] for i in range(t.shape[0]) if t.use_hrd[i]] #list of dataframes, one for each subject
        df = pd.concat(list_of_df) 
        df.to_csv(f'{temp_folder}\\outcomes_myhrd_df.csv')        
    if load_t:
        t = pd.read_csv(f'{temp_folder}\\outcomes_myhrd.csv')
    else:
        for i in range(t.shape[0]): #t.shape[0]
            if t.use_hrd[i]:
                print(subs[i])
                outcomes = amyhrd_utils.get_outcomes(subs[i],to_print_subject,to_plot_subject)                
                #save dictionary 'outcomes' to file
                import pickle
                with open(f'{temp_folder}\\outcomes_myhrd\\{subs[i]}_outcomes.pkl', 'wb') as f:
                    pickle.dump(outcomes, f)

                for cond in ['Intero','Extero']:
                    for j in outcomes[cond].keys():
                        t.loc[i,f'hrd_{cond}_{j}']=outcomes[cond][j]
        t.to_csv(f'{temp_folder}\\outcomes_myhrd.csv')
    print(f'Done loading data at {clock.time()[1]}')


    #Make a new group column with two groups (hc and PT) in both t and df
    t['group03'] = '' 
    df['group03'] = '' 
    for i,row in t.iterrows():
        record_id = row.record_id
        if hc[i]: 
            t.at[i,'group03'] = 'hc'
            df.loc[df.Subject==record_id,'group03'] = 'hc'
        elif eval(PT)[i]: 
            t.at[i,'group03'] = PT
            df.loc[df.Subject==record_id,'group03'] = PT
    #df = df[(df.group03=='hc') | (df.group03==PT)] #only include subjects in the two groups of interest


    #Fitting metacognition at group level with MLE. Based on https://colab.research.google.com/github/embodied-computation-group/metadpy/blob/master/docs/source/examples/Example%201%20-%20Fitting%20MLE%20-%20Subject%20and%20group%20level.ipynb#scrollTo=recognized-testament
    """
    from metadpy.mle import metad
    from metadpy.utils import discreteRatings
    print(f'Metacognition group level MLE start: {clock.time()[1]}')
    pd.options.mode.chained_assignment = None  # default='warn' 
    use_hrd_df = df.Subject.isin(t.record_id[t.use_hrd])
    this_df = df[(df.group03.isin(['hc',PT])) & (use_hrd_df) & (df.HeartRateOutlier==False) & (~df.Confidence.isnull())]
    new_confidence, _ = discreteRatings(this_df.Confidence) # discretize confidence ratings into 4 bins
    this_df['Confidence'] = new_confidence
    this_df['Stimuli'] = (this_df.Alpha > 0).astype('int')
    this_df['Responses'] = (this_df.Decision == 'More').astype('int')
    this_df['Accuracy']= this_df['Stimuli']==this_df['Responses']
    group_fit = metad(nRatings=4, stimuli="Stimuli", accuracy="Accuracy", confidence="Confidence", subject="Subject", padding=True, between="group03", within="Modality",data=this_df[confidence_all_options_df],verbose=True)   
    for i in range(len(group_fit)): #Enter values into dataframe t
        series = group_fit.iloc[i]
        modality = series['Modality']
        record_id = series['Subject']
        meta_d_groupMLE = series['meta_d']
        m_ratio_groupMLE = series['m_ratio']
        t.loc[t.record_id==record_id,f'hrd_{modality}_meta_d_groupMLE'] = meta_d_groupMLE
        t.loc[t.record_id==record_id,f'hrd_{modality}_m_ratio_groupMLE'] = m_ratio_groupMLE
    print(f'Metacognition group level MLE done: {clock.time()[1]}')
    """

    """
    #Fitting psychometric function group level, based on https://colab.research.google.com/github/embodied-computation-group/Cardioception/blob/master/docs/source/examples/psychophysics/2-psychophysics_group_level.ipynb#scrollTo=FOedFUWQcWHc
    import pytensor.tensor as pt
    import arviz as az
    import pymc as pm
    this_df = df[(df.HeartRateOutlier==False) & (df.Modality=='Intero')]
    this_df = this_df[['Alpha', 'Decision','Subject']]
    nsubj = this_df.Subject.nunique()
    x_total, n_total, r_total, sub_total = [], [], [], []
    for i, sub in enumerate(this_df.Subject.unique()):
        sub_df = this_df[this_df.Subject==sub]
        x, n, r = np.zeros(163), np.zeros(163), np.zeros(163)
        for ii, intensity in enumerate(np.arange(-40.5, 41, 0.5)):
            x[ii] = intensity
            n[ii] = sum(sub_df.Alpha == intensity)
            r[ii] = sum((sub_df.Alpha == intensity) & (sub_df.Decision == "More"))
        # remove no responses trials
        validmask = n != 0
        xij, nij, rij = x[validmask], n[validmask], r[validmask]
        sub_vec = [i] * len(xij)
        x_total.extend(xij)
        n_total.extend(nij)
        r_total.extend(rij)
        sub_total.extend(sub_vec)
    with pm.Model() as group_psychophysics:
        mu_alpha = pm.Uniform("mu_alpha", lower=-50, upper=50)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=100)
        mu_beta = pm.Uniform("mu_beta", lower=0, upper=100)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=100)
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=nsubj)
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=nsubj)
        thetaij = pm.Deterministic(
            "thetaij", amyhrd_utils.cumulative_normal(x_total, alpha[sub_total], beta[sub_total])
        )
        rij_ = pm.Binomial("rij", p=thetaij, n=n_total, observed=r_total)
    with group_psychophysics:
        idata = pm.sample(chains=4, cores=10)
    az.plot_trace(idata, var_names=["mu_alpha", "alpha"])
    stats = az.summary(idata, var_names=["mu_alpha", "mu_beta"])
    print(stats.loc['mu_alpha','mean'])
    print(stats.loc['mu_alpha','hdi_3%'])
    print(stats.loc['mu_alpha','hdi_97%'])
    """


    #Get confidence rating outliers
    Intero_confidence_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_Q_confidence_occurence_max'] > 0.65
    Extero_confidence_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_confidence_occurence_max'] > 0.65
    confidence_out = Intero_confidence_out | Extero_confidence_out
    confidence_not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & (hc|eval(PT)),'record_id'],confidence_out)

    has_sdt = ~t.hrd_Intero_dprime.isnull()
    has_metad_MLE = ~t.hrd_Intero_meta_d_MLE.isnull() & ~t.hrd_Extero_meta_d_MLE.isnull() & confidence_not_outliers
    has_metad_Bay = ~t.hrd_Intero_meta_d_Bay.isnull() & ~t.hrd_Extero_meta_d_Bay.isnull() & confidence_not_outliers

    for cond in ['Intero','Extero']: #which version of each outcome measure to use
        t[f'hrd_{cond}_threshold'] = t[f'hrd_{cond}_threshold_{psychometric_type}'] #options psi, Bay
        t[f'hrd_{cond}_slope'] = t[f'hrd_{cond}_slope_{psychometric_type}'] #options psi, Bay
        t[f'hrd_{cond}_meta_d'] = t[f'hrd_{cond}_meta_d_{metacog_type}'] #options MLE, Bay
        t[f'hrd_{cond}_m_ratio'] = t[f'hrd_{cond}_m_ratio_{metacog_type}'] #options MLE, Bay
    if metacog_type == 'MLE':
        has_metad = has_metad_MLE
    elif metacog_type == 'Bay':
        has_metad = has_metad_Bay


    """
    r: Outer level keys are 'Intero', 'Extero'. Inner level keys are quality measures (Q_wrong_decisions_took_longer, Q_wrong_decisions_lower_confidence, Q_confidence_occurence_max, Q_HR_outlier_perc), SDT measures (dprime, criterion), psychophysics measures (threshold, slope), HR measures (bpm_mean, RR_std,RMSSD) (HR not present for Extero condition)
    """

    """
    print('Optional: Convert psychometric measures to a percentage of subject-specific HR. Multiply by 70 so the result is in the range of BPM values')
    def divided_by_HR(column):
        #In dataframe t, replace divide values in column by the subject's HR. Then multiply all subjects' values by a common number (mean HR across all subjects).
        t.loc[t.use_hrd,column] = t.loc[t.use_hrd,'hrd_Intero_bpm_mean'].mean() * t.loc[t.use_hrd,column] / t.loc[t.use_hrd,'hrd_Intero_bpm_mean']
    divided_by_HR('hrd_Intero_threshold')
    divided_by_HR('hrd_Extero_threshold')
    divided_by_HR('hrd_Intero_slope')
    divided_by_HR('hrd_Extero_slope')
    """

    if outlier_method=='zscore':
        Extero_zscores = zscore(t.hrd_Extero_threshold,nan_policy='omit')
        Intero_zscores = zscore(t.hrd_Intero_threshold,nan_policy='omit')
        HR_zscores = zscore(t.hrd_Intero_bpm_mean,nan_policy='omit')
        not_outliers = (np.abs(Extero_zscores)<outlier_cutoff) & (np.abs(Intero_zscores)<outlier_cutoff) #define outliers as being more than 3 standard deviations from the mean for exteroceptive or interoceptive thresholds. 3/84 outliers leaving 81
    elif outlier_method=='madmedianrule':
        """
        Extero_out=pg.madmedianrule(t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_threshold']) 
        Intero_out=pg.madmedianrule(t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_threshold']) 
        HR_out=pg.madmedianrule(t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_bpm_mean']) 
        madmedian_outliers = Extero_out | Intero_out | HR_out
        not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & (hc|eval(PT)),'record_id'],madmedian_outliers) #13/84 outliers leaving 71
        """

        Extero_hc_out = pg.madmedianrule(t.loc[t.use_hrd & hc,'hrd_Extero_threshold']) #2 out
        Extero_PT_out = pg.madmedianrule(t.loc[t.use_hrd & eval(PT),'hrd_Extero_threshold']) #6 out
        Intero_hc_out = pg.madmedianrule(t.loc[t.use_hrd & hc,'hrd_Intero_threshold']) #1 out
        Intero_PT_out = pg.madmedianrule(t.loc[t.use_hrd & eval(PT),'hrd_Intero_threshold']) #1 out
        hc_out = Extero_hc_out | Intero_hc_out
        PT_out = Extero_PT_out | Intero_PT_out
        hc_not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & hc,'record_id'],hc_out) 
        PT_not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & eval(PT),'record_id'],PT_out)
        not_outliers = hc_not_outliers & PT_not_outliers #10/84 outliers leaving 74
        print(f'There are {(~not_outliers).sum()} outlier subjects')


    """
    #show the thresholds of subjects including outlier subjects
    t['not_outliers'] = not_outliers
    for column in ['hrd_Intero_threshold','hrd_Extero_threshold']:
        subgroup1='hc'
        subgroup2=PT
        column='hrd_Intero_threshold'
        x = t.loc[t.use_hrd & eval(subgroup1), column]
        y = t.loc[t.use_hrd & eval(subgroup2), column]     
        fig, ax = plt.subplots()
        sns.set_context('talk')
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sns.stripplot(ax=ax,y='group03',x=column,data=t.loc[t.use_hrd & (eval(subgroup1)|eval(subgroup2)),:],alpha=0.5,hue='not_outliers')
    plt.show()
    assert(0)
    """




    t.use_hrd = t.use_hrd & not_outliers #to exclude outliers from analyses

    print(f'hc, n={sum(t.use_hrd & hc)}')
    print(f'cc, n={sum(t.use_hrd & cc)}')
    print(f'sz, n={sum(t.use_hrd & sz)}')
    print(f'PT, n={sum(t.use_hrd & eval(PT))}')


    #Print quality checks (optional)
    """
    for cond in ['Intero','Extero']:
        for measure in ['wrong_decisions_took_longer','wrong_decisions_lower_alpha_pos','wrong_decisions_lower_alpha_neg','wrong_decisions_lower_confidence','confidence_occurence_max']:
            string = f'hrd_{cond}_Q_{measure}'
            values = t.loc[t.use_hrd,string]
            print(f'{cond} {measure} \t {np.mean(values):.2f}')
    values = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Intero_Q_HR_outlier_perc']
    print(f'Intero HR outlier perc \t {np.mean(values):.2f}')
    """

    #Get other quality check measures (optional)
    """
    Q1_Extero_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_wrong_decisions_took_longer'] == False
    Q2_Extero_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_wrong_decisions_lower_alpha_pos'] == False
    Q3_Extero_out = t.loc[t.use_hrd & (hc|eval(PT)),'hrd_Extero_Q_wrong_decisions_lower_alpha_neg'] == False
    Q_Extero_out = Q1_Extero_out | Q2_Extero_out | Q3_Extero_out
    Q_Extero_not_outliers = amyhrd_utils.get_outliers(t,t.loc[t.use_hrd & (hc|eval(PT)),'record_id'],Q_Extero_out)
    """
   
    #t.loc[t.use_hrd,'hrd_Intero_threshold'] = 70 * t.loc[t.use_hrd,'hrd_Intero_threshold'] / t.loc[t.use_hrd,'hrd_Intero_bpm_mean']

    #Use robust regression to predict interoceptive thresholds from exteroceptive thresholds
    x = t.loc[t.use_hrd & not_outliers,'hrd_Extero_threshold'] #'hrd_Extero_threshold'
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
    elif intero_adjust_method=='none':
        t['hrd_Intero_threshold_adj'] = t['hrd_Intero_threshold']

    #Absolute value of thresholds
    t['hrd_Intero_threshold_abs'] = np.abs(t.hrd_Intero_threshold)
    t['hrd_Extero_threshold_abs'] = np.abs(t.hrd_Extero_threshold)
    t['hrd_Intero_threshold_adj_abs'] = np.abs(t.hrd_Intero_threshold_adj)

    #print('Optional: Find the relationship between HR and adjusted interoceptive |threshold| in controls alone. Regress this relationship out of all adjusted interoceptive |thresholds|')
    """
    x = t.loc[t.use_hrd & not_outliers & hc, 'hrd_Intero_bpm_mean']
    y = t.loc[t.use_hrd & not_outliers & hc, 'hrd_Intero_threshold_adj_abs']
    reg = TheilSenRegressor().fit(x.values.reshape(-1,1),y.values)
    ynew = reg.predict(xnew.reshape(-1,1))
    fig,ax=plt.subplots()
    ax.scatter(x,y)
    ax.plot(xnew,ynew)
    ax.set_xlabel('BPM')
    ax.set_ylabel('hrd_Intero_threshold_adj_abs')
    ax.set_title(f'Theil-Sen regression: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}')
    t.loc[t.use_hrd,'hrd_Intero_threshold_adj_abs'] = t.loc[t.use_hrd,'hrd_Intero_threshold_adj_abs'] - reg.predict(t.loc[t.use_hrd,'hrd_Intero_bpm_mean'].values.reshape(-1,1)) 
    """

    #### PAIRWISE PLOTS #####
    
    #Correlations between HR measures. All highly correlated measures
    #grid=acommonfuncs.pairplot(t,vars=['hrd_Intero_bpm_mean','hrd_Intero_RR_std','hrd_Intero_RMSSD','meds_chlor'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

    #Correlations between psychophysics measures. Sig: Intero_threshold corr with Extero_threshold. Distributions markedly different across groups. Extero slope greater in patients.
    #grid=acommonfuncs.pairplot(t,['hrd_Intero_threshold','hrd_Intero_slope','hrd_Extero_threshold','hrd_Extero_slope'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)
    #Different versions of the 'threshold'. Only sig is extero_threshold_abs correlates with extero_slope
    #grid=acommonfuncs.pairplot(t,x_vars=['hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Extero_threshold_abs'],y_vars=['hrd_Intero_slope','hrd_Extero_threshold','hrd_Extero_slope'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

    #Psychophysics vs HR measures? Only hrd_Intero_threshold_adj_abs vs hrd_Intero_bpm_mean is significant (for regress_extero)
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Intero_slope'],y_vars = ['hrd_Intero_bpm_mean','hrd_Intero_RR_std'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Extero_threshold','hrd_Extero_threshold_abs','hrd_Extero_slope'],y_vars = ['hrd_Intero_bpm_mean','hrd_Intero_RR_std'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

    #Psychophysics vs clinical measures? Sig: In HC only, high IQ correlates with high Intero_threshold. In patients only, low IQ correlates with exteroceptive imprecision and high exteroceptive |threshold|, but not with interoceptive measures
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Intero_slope'],y_vars = ['fsiq2','sofas','panss_N'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Extero_threshold','hrd_Extero_threshold_abs','hrd_Extero_slope'],y_vars = ['fsiq2','sofas','panss_N'],include_these=t.use_hrd & not_outliers, height=1.0, robust=robust)

    #Psychophysics measures vs metacognition. Bayesian measures. High intero_threshold_abs corr with low Intero_meta_d
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Intero_threshold','hrd_Intero_threshold_abs','hrd_Intero_threshold_adj','hrd_Intero_threshold_adj_abs','hrd_Intero_slope'],y_vars = ['hrd_Intero_meta_d','hrd_Intero_m_ratio'],include_these=t.use_hrd & not_outliers & has_metad, height=1.0, robust=robust)
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Extero_threshold','hrd_Extero_threshold_abs','hrd_Extero_slope'],y_vars = ['hrd_Extero_meta_d','hrd_Extero_m_ratio'],include_these=t.use_hrd & not_outliers & has_metad, height=1.0, robust=robust)

    #Metacognition vs HR measures. No sig.
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Intero_meta_d','hrd_Intero_m_ratio'],y_vars = ['hrd_Intero_bpm_mean','hrd_Intero_RR_std'],include_these=t.use_hrd & has_metad, height=1.0, robust=robust)

 
    #Metacognition vs clinical measures. No sig.
    #grid=acommonfuncs.pairplot(t,x_vars = ['hrd_Intero_meta_d','hrd_Intero_m_ratio','hrd_Extero_meta_d','hrd_Extero_m_ratio'],y_vars = ['fsiq2','sofas','panss_N'],include_these=t.use_hrd & has_metad, height=1.0, robust=robust)
    

    #### SCATTERPLOTS OF 2 VARIABLES ####

    """
    #psi final vs fitted Bayesian estimates of psychophysics measures. Highly correlated
    for outcome in ['threshold','slope']: 
        for cond in ['Intero','Extero']:
            amyhrd_utils.scatter(t,'hc',PT,f'hrd_{cond}_{outcome}_psi',f'hrd_{cond}_{outcome}_Bay',robust=robust)

    #MLE vs Bayesian versions of metacognitive measures are highly correlated
    for outcome in ['meta_d','m_ratio']:
        for cond in ['Intero','Extero']:
            amyhrd_utils.scatter(t,'hc',PT,f'hrd_{cond}_{outcome}_MLE',f'hrd_{cond}_{outcome}_Bay',robust=robust,include_these = has_metad_MLE & has_metad_Bay)
    """
    #Significant results from pairplots
    amyhrd_utils.scatter(t,'hc',PT,'hrd_Intero_threshold','hrd_Extero_threshold',robust=robust)
    amyhrd_utils.scatter(t,'hc',PT,'hrd_Extero_threshold_abs','hrd_Extero_slope',robust=robust)
    amyhrd_utils.scatter(t,'hc',PT,'hrd_Intero_bpm_mean','hrd_Intero_threshold_adj',robust=robust)
    amyhrd_utils.scatter(t,'hc',PT,'hrd_Intero_bpm_mean','hrd_Intero_threshold_adj_abs',robust=robust)
    amyhrd_utils.scatter(t,'hc',PT,'fsiq2','hrd_Extero_threshold_abs',robust=robust)
    amyhrd_utils.scatter(t,'hc',PT,'fsiq2','hrd_Extero_slope',robust=robust)

    #PLOT DIFFERENCES ACROSS GROUPS
    for cond in ['Intero','Extero']:
        amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold',to_plot_compare=to_plot)
        amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold_abs')  
        if cond=='Intero':
            amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold_adj')
            amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_threshold_adj_abs')
        amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_slope',to_plot_compare=to_plot)
        amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_meta_d',include_these=has_metad,to_plot_compare=to_plot)
        amyhrd_utils.compare(t,'hc',PT,f'hrd_{cond}_m_ratio',include_these=has_metad,to_plot_compare=to_plot)



    #Replace interoceptive threshold with 'cardiac beliefs' (subject's own HR + interoceptive threshold)
    t['hrd_Intero_threshold_plus_HR'] = t.hrd_Intero_bpm_mean + t.hrd_Intero_threshold_adj
    x = t.loc[t.use_hrd & hc, 'hrd_Intero_threshold_plus_HR']
    y = t.loc[t.use_hrd & eval(PT), 'hrd_Intero_threshold_plus_HR']     
    mean_diff = np.mean(x)-np.mean(y)
    from scipy.stats import mannwhitneyu, ttest_ind
    p_ttest = ttest_ind(x,y).pvalue
    p_MW = mannwhitneyu(x,y).pvalue
    fig, ax = plt.subplots()
    sns.set_context('talk')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sns.stripplot(ax=ax,y='group03',x='hrd_Intero_threshold_plus_HR',data=t.loc[t.use_hrd & (eval('hc')|eval(PT)),:],alpha=0.5,palette=colors)
    ax.set_title(f'meandiff={mean_diff:.2f}, ttest p={p_ttest:.2f}, MW p={p_MW:.2f}')
    fig.tight_layout()
    sns.despine()



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
        #But HR doesn't account for interoceptive thresholds in controls if you look at plot of 'bpm' vs 'intero_threshold_adj_abs'


    elif intero_adjust_method in ['add7','none']:
        import warnings
        warnings.simplefilter("ignore", category=FutureWarning)
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
        warnings.simplefilter("default", category=FutureWarning)

        #pingoin's mixed_anova
        #print('Mixed ANOVA. hrd_threshold_abs ~ cond (within) + group03 (between) + cond*group03')
        #print(pg.mixed_anova(data=t2, dv='hrd_threshold_abs', within='cond', subject='record_id', between='group03')) #sig
        print('Mixed ANOVA. hrd_threshold_adj_abs ~ cond (within) + group03 (between) + cond*group03')
        print(pg.mixed_anova(data=t2, dv='hrd_threshold_adj_abs', within='cond', subject='record_id', between='group03',correction=True)) #not sig

        #As above but ANOVAs
        #print('ANOVA: hrd_Intero_threshold_adj_abs ~ group03 + hrd_Intero_bpm_mean')
        #print(sm.stats.anova_lm(smf.ols('hrd_Intero_threshold_adj_abs ~ group03 + hrd_Intero_bpm_mean', data=t.loc[t.use_hrd & (hc|eval(PT)),:]).fit(), typ=2))

        print('ANOVA: hrd_threshold_adj_abs ~ group03 + cond + group03:cond')
        print(sm.stats.anova_lm(smf.ols('hrd_threshold_adj_abs ~ group03 + cond + group03:cond', data=t2).fit(), typ=2)) #sig with sz, adj
        print('ANOVA: hrd_threshold_adj_abs ~ group03 + cond + group03:cond+hrd_Intero_bpm_mean')
        print(sm.stats.anova_lm(smf.ols('hrd_threshold_adj_abs ~ group03 + cond + group03:cond + hrd_Intero_bpm_mean', data=t2).fit(), typ=2)) 

        print('ANOVA: hrd_threshold_adj_abs ~ hrd_slope + cond + hrd_slope:cond')
        print(sm.stats.anova_lm(smf.ols('hrd_threshold_adj_abs ~ hrd_slope + cond + hrd_slope:cond', data=t2).fit(), typ=2)) 
        print("hrd_threshold_adj_abs ~ hrd_slope + cond + group03 + hrd_slope*cond + group03*cond + group03*hrd_slope + group03*cond*hrd_slope")
        print(sm.stats.anova_lm(smf.ols("hrd_threshold_adj_abs ~ hrd_slope + cond + group03 + hrd_slope*cond + group03*cond + group03*hrd_slope + group03*cond*hrd_slope", data=t2).fit(), typ=2)) #3-way. sig with adj
        print('ANOVA: hrd_slope ~ group03 + cond + group03:cond')
        print(sm.stats.anova_lm(smf.ols('hrd_slope ~ group03 + cond + group03:cond', data=t2).fit(), typ=2))

        #As above, but using random intercepts for subject
        # hrd_threshold_adj_abs ~ cond + group + group*cond + (1|record_id)
        mdf=smf.mixedlm("hrd_threshold_adj_abs ~ cond+group03+cond*group03", t2, groups=t2["record_id"]).fit().summary()
        mdf=smf.mixedlm("hrd_threshold_adj_abs ~ cond+group03+cond*group03+hrd_Intero_bpm_mean", t2, groups=t2["record_id"],re_formula="").fit().summary()

        mdf=smf.mixedlm('hrd_threshold_adj_abs ~ hrd_slope+cond+hrd_slope*cond',t2,groups=t2['record_id']).fit().summary()
        mdf=smf.mixedlm("hrd_threshold_adj_abs ~ cond+group03+hrd_slope+cond*group03+cond*hrd_slope+group03*hrd_slope+cond*group03*hrd_slope", t2, groups=t2["record_id"]).fit().summary()
        mdf=smf.mixedlm('hrd_slope ~ group03+cond+group03:cond',t2,groups=t2['record_id']).fit().summary()


        fig,axs=plt.subplots()
        sns.stripplot(data=t2,x='cond',y='hrd_threshold_adj',hue='group03',ax=axs,palette=colors)
        fig.tight_layout()


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