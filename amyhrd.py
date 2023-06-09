"""
Anayse myHRD
Basically Copied myJupHeartRateDiscrimination.ipynb
Needs conda environment 'hr'
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import numpy as np
from metadPy import sdt
from metadPy.utils import trials2counts, discreteRatings
from metadPy.plotting import plot_confidence
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from systole.detection import oxi_peaks
import pingouin as pg
from glob import glob
import re

top_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw\\"
files_with_task=glob(f"{top_folder}\\PCNS_*_BL\\beh\\HRD*\\")
subjects=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task
subjects_to_exclude=['015','031'] #exclude these subjects
"""
015, 067: weird ZeroDivisionError
031: overconfident, and really weird results?? maybe data could be ok except for confidence ratings??
"""
subjects_with_task = [subject for subject in subjects if subject not in subjects_to_exclude]

subjects_with_task=['004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '015', '019', '020', '022', '025', '031', '036']

subject='053'


contents=glob(f"{top_folder}\\PCNS_{subject}_BL\\beh\\HRD*\\")
assert(len(contents)==1)
resultsFolder=contents[0]

df=pd.read_csv(glob(f"{resultsFolder}*final.txt")[0]) # Logs dataframe
interoPost=np.load(glob(f"{resultsFolder}*Intero_posterior.npy")[0]) # History of posteriors distribution
exteroPost=np.load(glob(f"{resultsFolder}*Extero_posterior.npy")[0])
signal_df=pd.read_csv(glob(f"{resultsFolder}*signal.txt")[0]) # PPG signal

signal_df['Time'] = np.arange(0, len(signal_df))/1000 # Create time vector

df_old=df
df = df[df.RatingProvided == 1]

this_df = df[df.Modality=='Intero']
sum(this_df.TrialType == 'psi'), sum(this_df.TrialType == 'CatchTrial'), sum(this_df.TrialType == 'UpDown')

### Response time
sns.set_context('talk')
palette = ['#b55d60', '#5f9e6e']

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for i, task, title in zip([0, 1], ['DecisionRT', 'ConfidenceRT'], ['Decision', 'Confidence']):
    sns.boxplot(data=df, x='Modality', y='DecisionRT', hue='ResponseCorrect',
                palette=palette, width=.15, notch=True, ax=axs[i])
    sns.stripplot(data=df, x='Modality', y='DecisionRT', hue='ResponseCorrect',
                  dodge=True, linewidth=1, size=6, palette=palette, alpha=.6, ax=axs[i])
    axs[i].set_title(title)
    axs[i].set_ylabel('Response Time (s)')
    axs[i].set_xlabel('')
    axs[i].get_legend().remove()
sns.despine(trim=10)

handles, labels = axs[0].get_legend_handles_labels()
plt.legend(handles[0:2], ['Incorrect', 'Correct'], bbox_to_anchor=(1.05, .5), loc=2, borderaxespad=0.)
#plt.show()

### Metacognition
##SDT estimate for decision 1 perforamces (d' and criterion)

for i, cond in enumerate(['Intero', 'Extero']):
    this_df = df[df.Modality == cond].copy()
    this_df['Stimuli'] = (this_df.responseBPM > this_df.listenBPM)
    this_df['Responses'] = (this_df.Decision == 'More')

    hits, misses, fas, crs = sdt.scores(data=this_df)
    hr, far = sdt.rates(data=this_df,hits=hits, misses=misses, fas=fas, crs=crs)
    d, c = sdt.dprime(data=this_df,hit_rate=hr, fa_rate=far), sdt.criterion(data=this_df,hit_rate=hr, fa_rate=far)
    
    print(f'Condition: {cond} - d-prime: {d} - criterion: {c}')

sns.set_context('talk')
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

for i, cond in enumerate(['Intero', 'Extero']):
    this_df = df[df.Modality == cond]
    this_df = this_df[~this_df.Confidence.isnull()]
    new_confidence, _ = discreteRatings(this_df.Confidence)
    this_df['Confidence'] = new_confidence
    this_df['Stimuli'] = (this_df.Alpha > 0).astype('int')
    this_df['Responses'] = (this_df.Decision == 'More').astype('int')
    nR_S1, nR_S2 = trials2counts(data=this_df)
    plot_confidence(nR_S1, nR_S2, ax=axs[i])
    axs[i].set_title(f'{cond} condition')
sns.despine()
plt.tight_layout()
#plt.show()
#assert(0)
## Type 2 Psychometric function

sns.set_context('talk')
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
for i, modality, col in zip((0, 1), ['Extero', 'Intero'], ['#4c72b0', '#c44e52']):
    
    this_df = df[df.Modality == modality]
    
    # Plot data points
    for ii, intensity in enumerate(np.sort(this_df.Alpha.unique())):
        conf = this_df.Confidence[(this_df.Alpha == intensity)].mean()
        total = sum(this_df.Alpha == intensity)
        axs[i].plot(intensity, conf, 'o', alpha=0.5, color=col, markeredgecolor='k', markersize=total*3)
        axs[i].set_title(modality)
        axs[i].set_ylabel('Confidence rating')
        axs[i].set_xlabel('Intensity ($\Delta$ BPM)')
plt.tight_layout()
sns.despine()
#plt.show()

### Psychophysics
sns.set_context('talk')
fig, axs = plt.subplots(1, 1, figsize=(8, 5))

for cond, col in zip(['Intero', 'Extero'], ['#4c72b0', '#c44e52']):
    this_df = df[df.Modality == cond]
    axs.hist(this_df.Alpha, color=col, bins=np.arange(-40.5, 40.5, 5), histtype='stepfilled',
             ec="k", density=True, align='mid', label=cond, alpha=.6)
axs.set_title('Distribution of the tested intensities values')
axs.set_xlabel('Intensity (BPM)')
plt.legend()
sns.despine(trim=10)
plt.tight_layout()


#Updown staircase
fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
sns.set_context('talk')

for i, cond, col in zip([0, 1], ['Intero', 'Extero'], ['#c44e52', '#4c72b0']):
    this_df = df[(df.Modality == cond) & (df.TrialType == 'UpDown')]

    axs[i].plot(np.arange(0, len(this_df))[this_df.StairCond == 'high'], 
                    this_df.Alpha[this_df.StairCond == 'high'], linestyle='--', color=col, linewidth=2)
    axs[i].plot(np.arange(0, len(this_df))[this_df.StairCond == 'low'], 
                    this_df.Alpha[this_df.StairCond == 'low'], linestyle='-', color=col, linewidth=2)
    
    axs[i].plot(np.arange(0, len(this_df))[this_df.Decision == 'More'], 
                    this_df.Alpha[this_df.Decision == 'More'], col, marker='o', linestyle='', markeredgecolor='k', label=cond)
    axs[i].plot(np.arange(0, len(this_df))[this_df.Decision == 'Less'], 
                    this_df.Alpha[this_df.Decision == 'Less'], 'w', marker='s', linestyle='', markeredgecolor=col, label=cond)

    axs[i].axhline(y=0, linestyle='--', color = 'gray')
    handles, labels = axs[i].get_legend_handles_labels()
    axs[i].legend(handles[0:2], ['More', 'Less'], borderaxespad=0., title='Decision')
    axs[i].set_ylabel('Intensity ($\Delta$ BPM)')
    axs[i].set_xlabel('Trials')
    axs[i].set_ylim(-42, 42)
    axs[i].set_title(cond+'ception')
    sns.despine(trim=10, ax=axs[i])
    plt.gcf()

#Psi staircase
fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
sns.set_context('talk')

# Plot confidence interval for each staircase
def ci(x):
    return np.where(np.cumsum(x) / np.sum(x) > .025)[0][0], \
           np.where(np.cumsum(x) / np.sum(x) < .975)[0][-1]

for i,cond, stair, col in zip([0, 1], ['Intero', 'Extero'], [interoPost, exteroPost], ['#c44e52', '#4c72b0']):
    this_df=df[(df.Modality==cond) & (df.TrialType == 'psi')] #excludes catchTrials 
    this_df_old=df_old[(df_old.Modality == cond) & (df_old.TrialType == 'psi')] #excludes catchTrials but includes missing
    nonMissingIndices=np.where(this_df_old['RatingProvided'] )[0] #indices of trials (in df_old) with Rating Provided
    stair_nonMissing=stair[nonMissingIndices,:,:]
    """
    print('i col:',i,col)
    print('this_df.shape',this_df.shape)
    print('this_df_old.shape',this_df_old.shape)
    print('stair old',stair.shape[0])
    print('stair new',stair_nonMissing.shape[0])
    """
    ciUp, ciLow = [], []
    for t in range(stair_nonMissing.shape[0]):
        #print(t)
        up, low = ci(stair_nonMissing.mean(2)[t])
        ciUp.append(np.arange(-50.5, 50.5)[up])
        ciLow.append(np.arange(-50.5, 50.5)[low])
    axs[i].fill_between(x=np.arange(len(this_df)),
                        y1=ciLow,
                        y2=ciUp,
                        color=col, alpha=.2)

# Staircase traces
for i, cond, col in zip([0, 1], ['Intero', 'Extero'], ['#c44e52', '#4c72b0']):
    this_df = df[(df.Modality == cond) & (df.TrialType != 'UpDown')]

    # Show UpDown staircase traces
    axs[i].plot(np.arange(0, len(this_df))[this_df.StairCond == 'high'], 
                    this_df.Alpha[this_df.StairCond == 'high'], linestyle='--', color=col, linewidth=2)
    axs[i].plot(np.arange(0, len(this_df))[this_df.StairCond == 'low'], 
                    this_df.Alpha[this_df.StairCond == 'low'], linestyle='-', color=col, linewidth=2)

    axs[i].plot(np.arange(0, len(this_df))[this_df.Decision == 'More'], 
                    this_df.Alpha[this_df.Decision == 'More'], col, marker='o', linestyle='', markeredgecolor='k', label=cond)
    axs[i].plot(np.arange(0, len(this_df))[this_df.Decision == 'Less'], 
                    this_df.Alpha[this_df.Decision == 'Less'], 'w', marker='s', linestyle='', markeredgecolor=col, label=cond)

    # Trheshold estimate
    axs[i].plot(np.arange(sum(this_df.StairCond != 'psi'), len(this_df)), this_df[this_df.StairCond == 'psi'].EstimatedThreshold, linestyle='-', color=col, linewidth=4)
    axs[i].plot(np.arange(0, sum(this_df.StairCond != 'psi')), this_df[this_df.StairCond != 'psi'].EstimatedThreshold, linestyle='--', color=col, linewidth=2, alpha=.3)

    axs[i].axhline(y=0, linestyle='--', color = 'gray')
    handles, labels = axs[i].get_legend_handles_labels()
    axs[i].legend(handles[0:2], ['More', 'Less'], borderaxespad=0., title='Decision')
    axs[i].set_ylabel('Intensity ($\Delta$ BPM)')
    axs[i].set_xlabel('Trials')
    axs[i].set_ylim(-42, 42)
    axs[i].set_title(cond+'ception')
    sns.despine(trim=10, ax=axs[i])
    plt.gcf()
    
###Psychometric function
sns.set_context('talk')
fig, axs = plt.subplots(figsize=(8, 5))
for i, modality, col in zip((0, 1), ['Extero', 'Intero'], ['#4c72b0', '#c44e52']):
    
    this_df = df[(df.Modality == modality) & (df.TrialType == 'psi')]

    t, s = this_df.EstimatedThreshold.iloc[-1], this_df.EstimatedSlope.iloc[-1]
    # Plot Psi estimate of psychometric function
    axs.plot(np.linspace(-40, 40, 500), 
            (norm.cdf(np.linspace(-40, 40, 500), loc=t, scale=s)),
            '--', color=col, label=modality)
    # Plot threshold
    axs.plot([t, t], [0, .5], color=col, linewidth=2)
    axs.plot(t, .5, 'o', color=col, markersize=10)

    # Plot data points
    for ii, intensity in enumerate(np.sort(this_df.Alpha.unique())):
        resp = sum((this_df.Alpha == intensity) & (this_df.Decision == 'More'))
        total = sum(this_df.Alpha == intensity)
        axs.plot(intensity, resp/total, 'o', alpha=0.5, color=col, markeredgecolor='k', markersize=total*5)
plt.ylabel('P$_{(Response = More|Intensity)}$')
plt.xlabel('Intensity ($\Delta$ BPM)')
plt.tight_layout()
plt.legend()
sns.despine()


#### Visualise PPG
drop, bpm_std, bpm_df = [], [], pd.DataFrame([])
clean_df = df.copy()
clean_df['HeartRateOutlier'] = np.zeros(len(clean_df), dtype='bool')
for i, trial in enumerate(signal_df.nTrial.unique()):
    color = '#c44e52' if (i % 2) == 0 else '#4c72b0'
    this_df = signal_df[signal_df.nTrial==trial]  # Downsample to save memory
    
    signal, peaks = oxi_peaks(this_df.signal, sfreq=1000)
    bpm = 60000/np.diff(np.where(peaks)[0])
    
    bpm_df = bpm_df.append(pd.DataFrame({'bpm': bpm, 'nEpoch': i, 'nTrial': trial}))

# Check for outliers in the absolute value of RR intervals 
for e, t in zip(bpm_df.nEpoch[pg.madmedianrule(bpm_df.bpm.to_numpy())].unique(),
                bpm_df.nTrial[pg.madmedianrule(bpm_df.bpm.to_numpy())].unique()):
    drop.append(e)
    clean_df.loc[t, 'HeartRateOutlier'] = True

# Check for outliers in the standard deviation values of RR intervals 
for e, t in zip(np.arange(0, bpm_df.nTrial.nunique())[pg.madmedianrule(bpm_df.copy().groupby(['nTrial', 'nEpoch']).bpm.std().to_numpy())],
                bpm_df.nTrial.unique()[pg.madmedianrule(bpm_df.copy().groupby(['nTrial', 'nEpoch']).bpm.std().to_numpy())]):
    if e not in drop:
        drop.append(e)
        clean_df.loc[t, 'HeartRateOutlier'] = True


meanBPM, stdBPM, rangeBPM = [], [], []

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(30, 10))
for i, trial in enumerate(signal_df.nTrial.unique()):
    
    color = '#c44e52' if (i % 2) == 0 else '#4c72b0'
    this_df = signal_df[signal_df.nTrial==trial]  # Downsample to save memory
    
    # Mark as outlier if relevant
    if i in drop:
        ax[0].axvspan(this_df.Time.iloc[0], this_df.Time.iloc[-1], alpha=.3, color='gray')
        ax[1].axvspan(this_df.Time.iloc[0], this_df.Time.iloc[-1], alpha=.3, color='gray')
    
    ax[0].plot(this_df.Time, this_df.signal, label='PPG', color=color, linewidth=.5)

    # Peaks detection
    signal, peaks = oxi_peaks(this_df.signal, sfreq=1000)
    bpm = 60000/np.diff(np.where(peaks)[0])
    m, s, r = bpm.mean(), bpm.std(), bpm.max() - bpm.min()
    meanBPM.append(m)
    stdBPM.append(s)
    rangeBPM.append(r)

    # Plot instantaneous heart rate
    ax[1].plot(this_df.Time.to_numpy()[np.where(peaks)[0][1:]], 
               60000/np.diff(np.where(peaks)[0]),
              'o-', color=color, alpha=0.6)

ax[1].set_xlabel("Time (s)")
ax[0].set_ylabel("PPG level (a.u.)")
ax[1].set_ylabel("Heart rate (BPM)")
ax[0].set_title("PPG signal recorded during interoceptive condition (5 seconds each)")
sns.despine()
ax[0].grid(True)
ax[1].grid(True)

### Heart rate summary statistics
sns.set_context('talk')
fig, axs = plt.subplots(figsize=(13, 5), nrows=2, ncols=2)
meanBPM = np.delete(np.array(meanBPM), np.array(drop))
stdBPM = np.delete(np.array(stdBPM), np.array(drop))
for i, metric, col in zip(range(3), [meanBPM, stdBPM], ['#b55d60', '#5f9e6e']):
    axs[i, 0].plot(metric, 'o-', color=col, alpha=.6)
    axs[i, 1].hist(metric, color=col, bins=15, ec="k", density=True, alpha=.6)
    axs[i, 0].set_ylabel('Mean BPM' if i == 0 else 'STD BPM')
    axs[i, 0].set_xlabel('Trials')
    axs[i, 1].set_xlabel('BPM')
sns.despine()
plt.tight_layout()

plt.show()