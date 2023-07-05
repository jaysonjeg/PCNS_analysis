"""
Analyse pupil data for cface task. Conditions are HAHA, HAAN, ANHA, ANAN.
Remember that the cface task was run twice, once in the MRI with eye tracking, and once outside the MRI with face recording. So we can't compare single-trial face data with single-trial eye data.

Pupil data preprocessed by Anna Behler. Removed blinks, resampled to 100Hz, smoothing. Some entire trials were excluded in preprocessing, and don't have columns in {subject}_cface_BaselineCorrPupil.csv
For each subject, get median time series for each condition (HAHA,HAAN,ANHA,ANAN)
For each subject and condition, outlier trials will have a large standard deviation. Use the MAD-median rule to exclude these.

Outlier subjects:
(1) abnormally large standard deviation of baseline pupil size across trials
(2) abnormal standard deviation of pupil values in the median trace for any of the 4 conditions (optional)

Optional: Look at percentage changes from baseline rather than absolute changes, so it is more comparable across subjects

To consider:
Anna: Normalise by dividing by baseline rather than subtracting?
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, pingouin as pg, seaborn as sns
import acommonfuncs, acface_utils
from acface_utils import ntrials, n_trialsperemotion, emots, relevant_timejitters, relevant_labels, relevant_timestamps, midpoint_timestamps
from acommonvars import *

c = acommonfuncs.clock()
t['use_cface'] = ((include) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #those subjects whose cface data we will use
t['use_cface_eye'] = t['use_cface'] & (t.valid_eyetracking==1)
eyedata_raw_folder = f'{analysis_folder}\\pupils_cface1'

### SETTABLE PARAMETERS 
group = 'group02' #the grouping variable
divide_pupil_by_baseline = False #for each trial, divide pupil values by baseline pupil size at the start of that trial
to_plot_subject = False

### Get some variables common to all participants
pupil_raw = pd.read_csv(f'{eyedata_raw_folder}\\035_cface_BaselineCorrPupil.csv')
times_trial_regular = pupil_raw.iloc[:,0].values #timestamps in seconds
df = acommonfuncs.get_beh_data('cface1','035','out',use_MRI_task=True) #behavioural data. The sequence of conditions (HAHA, HAAN, etc) is same for all participants, so we can get the sequence from any participant

conds = ['HAHA','HAAN','ANHA','ANAN']
conds_trials = {cond:np.where(df.type==cond)[0] + 1 for cond in conds} #dictionary of condition:trial numbers. Add 1 because rows of df have 0-based indexing, whereas trial numbers in the columns of pupil_raw are 1-based indexing
HAAN = np.where(df.type=="HAAN")[0] + 1
conds_trials_str = {cond:[str(i) for i in conds_trials[cond]] for cond in conds} #dictionary of condition:trial numbers as strings

#Optional check that all subjects have the same sequence of conditions
"""
for i in range(len(t)):
    if t['use_cface'][i] and t['valid_eyetracking'][i]==1:
        subject=t.subject[i]
        df = acommonfuncs.get_beh_data('cface1',subject,'out',use_MRI_task=True)
        HAHA_trialnumbers = np.where(df.type=='HAHA')[0] + 1
        print(HAHA_trialnumbers[-1])
"""

def remove_outlier_trials(df):
    standard_deviations = df.std(axis=0)
    outliers = pg.madmedianrule(standard_deviations)
    #print(f'Outlier trials: {np.sum(outliers)}/{len(df.columns)}')
    return df.loc[:,~outliers] 

#### Get median pupil time series for each condition in each subject, and put them into dataframe t
pupil_variables = ['cface_pupil_HAHA','cface_pupil_HAAN','cface_pupil_ANHA','cface_pupil_ANAN']
load_table=False
if load_table:
    t = pd.read_csv(f'{temp_folder}\\outcomes_cface_eye.csv')
    t = acommonfuncs.str_columns_to_literals(t,pupil_variables)
else:
    t=acommonfuncs.add_columns(t,pupil_variables)
    for i in range(len(t)):
        if t['use_cface_eye'][i]:
            subject=t.subject[i]
            #next line gets the file {eyedata_raw_folder}\\{subject}_cface_Baseline.csv
            baseline = pd.read_csv(f'{eyedata_raw_folder}\\{subject}_cface_Baseline.csv') #baseline pupil size for each trial. Note that there may not be a row for every single trial.
            pupil_raw = pd.read_csv(f'{eyedata_raw_folder}\\{subject}_cface_BaselineCorrPupil.csv') #column headings are actual trial number. Each column contains a time series of pupil size for that trial. Note that there may not be a column for every single trial.

            t.at[i,'cface_pupil_base_median'] = baseline.baseline_pupil.median()
            t.at[i,'cface_pupil_base_std'] = baseline.baseline_pupil.std()

            #Check whether for this subject, the baseline pupil size correlates with standard deviation of pupil size in the subsequent trial. 
            """
            trial_baseline_values = [baseline.loc[baseline.trial_number==trial,'baseline_pupil'].values[0] for trial in baseline.trial_number]
            trial_std = [pupil_raw.loc[:,str(trial)].std() for trial in baseline.trial_number]
            print(f'Correlation between baseline and trial std: {np.corrcoef(trial_baseline_values,trial_std)[0,1]:.2f}')
            """

            print(f'Subject {subject}',end=", ")

            pupil_allconds = {cond:[] for cond in conds}

            for cond in conds:
                #print(f'Subject {subject}, cond {cond}')
                which_trials = [i for i in conds_trials_str[cond] if i in pupil_raw.columns] #only include trials that have a column in pupil_raw

                pupil = pupil_raw.loc[:,which_trials]

                if divide_pupil_by_baseline:
                    which_trials_int = [int(i) for i in which_trials]
                    baselines = baseline.loc[baseline.trial_number.isin(which_trials_int),'baseline_pupil'].values 
                    pupil = pupil.div(baselines,axis=1)


                pupil = remove_outlier_trials(pupil)
                pupil_allconds[cond] = pupil

                pupil_std_mean = pupil.std(axis=1).mean() #mean standard deviation across trials
                t.at[i,f'cface_pupil_{cond}_std'] = pupil_std_mean

                pupil_median = pupil.median(axis=1).values.astype(np.float32) #median across trials
                t.at[i,f'cface_pupil_{cond}'] = list(pupil_median)
            
            if to_plot_subject or (int(subject) in []):
                plt.figure(figsize = [12, 8]) #Plot time series for each trial separately, colour coded by condition
                plt.plot(times_trial_regular,pupil_allconds['HAHA'],color='red',label='HAHA',linewidth=1,alpha=0.5)
                plt.plot(times_trial_regular,pupil_allconds['HAAN'],color='blue',label='HAAN',linewidth=1,alpha=0.5)
                plt.plot(times_trial_regular,pupil_allconds['ANHA'],color='green',label='ANHA',linewidth=1,alpha=0.5)
                plt.plot(times_trial_regular,pupil_allconds['ANAN'],color='black',label='ANAN',linewidth=1,alpha=0.5)
                plt.xlabel('Time (s)')
                plt.ylabel('Pupil size (a.u.)')
                plt.title(f'Subject {subject}')
                plt.show()


    t.to_csv(f'{temp_folder}\\outcomes_cface_eye.csv')

#Exclude subjects for whom the baseline pupil size varies a lot across trials (4 subs)
outliers_base_std_bool = pg.madmedianrule(t.loc[t.use_cface_eye,'cface_pupil_base_std']) 
outliers_base_std = t.loc[t.use_cface_eye,'subject'][outliers_base_std_bool].values
print(f'\nOutliers, std of baseline pupil size: {outliers_base_std}')
include_subjects_base_std = ~t.subject.isin(outliers_base_std)

#Exclude subjects whose median pupil trace for HAHA or HAAN, has a large standard deviation
def func(cond):
    #returns standard deviation of pupil size in each subject's median trace
    array= np.vstack(t.loc[t.use_cface_eye,f'cface_pupil_{cond}']).T
    array_std = array.std(axis=0)
    return array_std
dict_of_std = {cond: func(cond) for cond in conds}
outliers_std_bool = {cond: pg.madmedianrule(dict_of_std[cond]) for cond in conds}
outliers_std_bool_all = outliers_std_bool['HAHA'] | outliers_std_bool['HAAN']
outliers_std=t.loc[t.use_cface_eye,'subject'][outliers_std_bool_all].values
print(f'Outliers, std of median pupil trace for HAHA or HAAN: {outliers_std}')
include_subjects_std = ~t.subject.isin(outliers_std)

#Exclude subjects in a manually specified list

exclude_subjects_general = [25,29,35,42,46,47,52,66,67,69,70,95,105,124] #mostly due to frequent blinks, or blinks at end of trial
blinks_at_end = [35,73,100,108,113,122,130] #subjects who have a strong blink at end of each trial
exclude_subjects_AN = [25,33,51,55,57,60,71,72,76,79,84,85,86,87,117,121,130,133] #subjects whose frown trials must be excluded because their eyes semi-shut, changing pupil size
exclude_subjects_AN_borderline = [24,28]
exclude_subjects = exclude_subjects_general

print(f'Outliers, manually specified: {exclude_subjects}')
include_subjects = np.array([True]*len(t))
for subject in exclude_subjects: 
    include_subjects [t.record_id == subject] = False

def plot_one(ax,x,y):
    ax.scatter(dict_of_std[x][~outliers_std_bool_all],dict_of_std[y][~outliers_std_bool_all])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
fig,axs=plt.subplots(2,2)
plot_one(axs[0,0],'HAHA','HAAN')
plot_one(axs[0,1],'HAHA','ANHA')
plot_one(axs[1,0],'ANAN','HAAN')
plot_one(axs[1,1],'ANAN','ANHA')
fig.suptitle('Standard deviation of pupil size in each subject\'s median trace')
fig.tight_layout()


t.use_cface_eye = t.use_cface_eye & include_subjects & include_subjects_base_std #& include_subjects_std

fig,axs=plt.subplots(2)
for ax,column_name in zip(axs,['cface_pupil_HAHA','cface_pupil_HAAN']):
    print(f'{c.time()[1]}: Starting lineplot with bootstrapped CIs for group median trace for {column_name}')
    df = t.loc[t.use_cface_eye & (t[group]!=''),column_name]
    array = np.vstack(df).T
    subject_names = t.subject[t.use_cface_eye & (t[group]!='')].values
    df2 = pd.DataFrame(array, columns = subject_names )
    df2['time'] =  times_trial_regular
    df3 = pd.melt(df2, id_vars=['time'],value_vars = subject_names, var_name='subject', value_name='pupil size')
    df3[group]=''
    for gp in t[group].unique():
        df3.loc[df3.subject.isin(t.subject[t[group]==gp]).values,group] = gp
    sns.lineplot(df3,x='time',y='pupil size',hue=group,palette=colors,ax=ax,n_boot=500,estimator=np.median)
    ax.set_title(column_name)

def plot_pupil_subjects(ax,times_trial_regular,variable,group):
    for gp in t[group].unique():
        array= np.vstack(t.loc[t.use_cface_eye & (t[group]==gp),variable]).T
        ax.plot(times_trial_regular, array, color=colors[gp],alpha=0.3,linewidth=1)
        ax.plot(times_trial_regular, np.median(array,axis=1), color=colors[gp],alpha=0.3, linewidth=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pupil size (a.u.)')
        ax.set_title(variable)
fig,axs=plt.subplots(2,2)
plot_pupil_subjects(axs[0,0],times_trial_regular,'cface_pupil_HAHA',group)
plot_pupil_subjects(axs[0,1],times_trial_regular,'cface_pupil_HAAN',group)
plot_pupil_subjects(axs[1,0],times_trial_regular,'cface_pupil_ANHA',group)
plot_pupil_subjects(axs[1,1],times_trial_regular,'cface_pupil_ANAN',group)
fig.tight_layout()

z = t.loc[t.use_cface_eye,:]
fig,axs=plt.subplots(2,2)
sns.scatterplot(ax=axs[0,0],data=z,x='cface_pupil_base_std',y='cface_pupil_HAHA_std',hue=group,palette=colors)
sns.scatterplot(ax=axs[0,1],data=z,x='cface_pupil_base_median',y='cface_pupil_HAHA_std',hue=group,palette=colors)

sns.scatterplot(ax=axs[1,0],data=z,x='cface_pupil_base_std',y='cface_pupil_ANAN_std',hue=group,palette=colors)
sns.scatterplot(ax=axs[1,1],data=z,x='cface_pupil_base_median',y='cface_pupil_ANAN_std',hue=group,palette=colors)
fig.tight_layout()
#sns.stripplot(z.cface_pupil_base_std)

plt.show(block=False)