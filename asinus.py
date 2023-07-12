"""
Analyse Facial mirring task (sinus)
10 trials, each lasting 20s

Webcam recording received at 30fps, but task saved data (and stimulus display) are at 25 fps (0.04 sec/frame)

There are 3 trial types.
First 2 trials ('initial') are no feedback, predictable
Next 5 trials ('jump') are no feedback, pjump=[0.025,0.03,0.035,0.04,0.045]
Last 3 trials ('final') are feedback present, predictable

New columns
    use_sinus
    sinus_outliers
    blinks, amp, rapidity, lag_hilbert, lag_fourier (for these columns we have versions for trial types 'initial' and 'final')
    corrs, plv (for these columns we have versions for all 3 trial types) 

Outcomes: We find the following within each trial, then take the median across each trial type
    blinks: proportion of frames where blink was detected. Can be an index of tiredness
    amp: amplitude, the maximum intensity of AU12 within the trial
    rapidity: rapidity of onset of AU12, specifically the trialwise median of peak heights of gradient

    corrs: spearman correlation between stimulus and response AU12 time series
    plv: phase locking value: Is the relative phase constant across time?

    lag_hilbert: median phase difference between stimulus and response, calculated using Hilbert transform
    lag_fourier: median phase difference between stimulus and response, calculated using FFT

Questions
    Does amp and rapidity decrease from start to end?
    Corrs or PLV for jump trials might need to be normalised by the values for initial trials
"""


#To get AU intensities for stimulus face
"""
import subprocess
vids_folder = "D:\\FORSTORAGE\\MY_STIMULI\\for_sinus\\png1_Neutral-Happiness\\F01-NE-HA.mp4"
out_folder=f'{vids_folder}\OpenFace_static' #where to save
openfacefolder='D:/FORSTORAGE/OpenFace-master'
openfacefile=f'{openfacefolder}/OpenFace-master/x64/Release/FeatureExtraction.exe'
commands=[f'{openfacefile} -fdir {vids_folder} -au_static -out_dir {out_folder} -aus']
for command in commands:
    subprocess.call(command)
"""


import numpy as np, pandas as pd, seaborn as sns, pingouin as pg
import matplotlib.pyplot as plt
from glob import glob
import scipy.io
import warnings
from scipy import signal,stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from acommonvars import *
import acommonfuncs
import asinus_utils


def get_sinus_data(subject):
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\sinus*\\") 
    assert(len(contents)==1) 
    resultsFolder=contents[0]
    mat=scipy.io.loadmat(glob(f"{resultsFolder}*.mat")[0])
    ausdata = mat['ausdata'].squeeze() #10 trials * 500 timepoints, with each element being (n,17) array of AU intensities. n is usually 1, but can be 2 (if 2 samples received from webcam) or 0
    posdata = mat['posdata'].squeeze()
    metadata = mat['metadata'].squeeze()
    return ausdata,posdata,metadata


AU_to_plot = 'AU12'
aulabels_list=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45'] 
nAU = aulabels_list.index(AU_to_plot)


ntrials = 10 #number of trials per subject
nframes = 500 #number of frames per trial
time_interval = 0.04 #seconds per frame

fs = int(1/time_interval) #sampling frequency
nseconds_per_trial = nframes*time_interval
times_trial_regular = np.linspace(time_interval, nseconds_per_trial, nframes)

outliers=['009','016','018','066'] #exclude these subjects

def make_bool(true_indices,length):
    temp = np.zeros(length,dtype=bool)
    temp[true_indices]=True
    return temp
"""
Trial types
initial: trials without green screen feedback
jump: trials with random jumps
final: trials with green screen feedback
"""
trials = {'initial':[0,1],'jump':[2,3,4,5,6],'final':[7,8,9]}
trials_bool = {key:make_bool(trials[key],10) for key in trials.keys()}

_,posdata,metadata = get_sinus_data('020') #posdata and metadata are same for every subject

stimulus = pd.read_csv(f'{analysis_folder}\\sinus\\F01-NE-HA_OpenFace_static.csv')[f' {AU_to_plot}_r']

posdata=np.vectorize(lambda x: stimulus[x])(posdata) #pos data contains which frame number of stimulus face (0 to 30) was shown at each time point. Convert these to stimulus AU intensity values

if __name__=='__main__':
    c = acommonfuncs.clock()

    ### SETTABLE PARAMETERS 
    group = 'group02' #the grouping variable
    load_table=True

    ### Get data
    new_columns = ['use_sinus','sinus_outliers']
    for string in ['blinks', 'amp', 'rapidity', 'lag_hilbert', 'lag_fourier']:
        new_columns.append(f'sinus_{string}_initial')
        new_columns.append(f'sinus_{string}_final')
    for string in ['corrs', 'plv']:
        new_columns.append(f'sinus_{string}_initial')
        new_columns.append(f'sinus_{string}_jump')
        new_columns.append(f'sinus_{string}_final')

    if load_table:
        t = acommonfuncs.add_table(t,'outcomes_sinus.csv')
        #t = acommonfuncs.str_columns_to_literals(t,['sinus_ts'])
    else:
        t['use_sinus'] = ((include) & (t.valid_sinuso==1)) 
        t['sinus_outliers'] = t.subject.isin(outliers)
        #t=acommonfuncs.add_columns(t,['sinus_ts'])
        for t_index in range(len(t)):
            if (t['use_sinus'][t_index]) and (t.subject[t_index] not in outliers):
                subject=t.subject[t_index]
                print(f'{c.time()[1]}: Subject {subject}')
                ausdata,_,_ = get_sinus_data(subject)

                #In each frame, find the mean (if there are more than 1 webcame frames)
                for i in range(ausdata.shape[0]):
                    for j in range(ausdata.shape[1]):
                        if len(ausdata[i,j])!=0:
                            ausdata[i,j] = ausdata[i,j].mean(axis=0)
                        else:
                            temp = np.array([np.nan]*len(aulabels_list))
                            ausdata[i,j]=temp

                #Interpolate missing timepoints, and reshape into 3D array
                temp = []
                for trial in range(ntrials):
                    ausdata_trial = np.vstack(ausdata[trial,:])
                    temp.append(pd.DataFrame(ausdata_trial).interpolate(method='linear',axis=0,limit_direction='both').values) 

                aus = np.transpose(np.dstack(temp) , axes=[2,0,1]) #ntrials (10) * nframes (500) * nAUs (17)
                blinks = np.array([np.sum(aus[trial,-1,:]>0)/nframes for trial in range(ntrials)]) #proportion of blinks
                amp = np.array([np.max(aus[trial,:,nAU]) for trial in range(ntrials)]) #max intensity of AU12

                #Normalize each trial's time series to range (0,1)
                from sklearn import preprocessing 
                posdata = preprocessing.minmax_scale(posdata, feature_range=(0, 1), axis=1, copy=True)
                aus2 = np.transpose(aus,axes=[0,2,1])
                aus3 = preprocessing.minmax_scale(aus2.reshape(-1,aus2.shape[-1]), feature_range=(0, 1), axis=1, copy=True).reshape(aus2.shape)
                ausn = np.transpose(aus3, axes=[0,2,1])

                #low-pass filter (use filtered data for rapidity and hilbert transform)
                cutoff = 3
                posdatas = asinus_utils.lowpass(posdata,cutoff)
                ausns = asinus_utils.lowpass(ausn,cutoff,axis=1) 
                #posdatas=posdata   
                #ausns = ausn         

                #Rapidity: Get gradient of (0-1)-normalized then smoothed AU time series. Find peaks in the gradient. Then find the median of the peak height. This measures the rapidity of the person's facial response. 
                ausns_grad_peaks_initial = asinus_utils.get_grad_peak_heights(ausns[trials_bool['initial'],:,nAU])
                rapidity_initial = np.median(ausns_grad_peaks_initial)
                ausns_grad_peaks_final = asinus_utils.get_grad_peak_heights(ausns[trials_bool['final'],:,nAU])
                rapidity_final = np.median(ausns_grad_peaks_final)

                #Spearman correlation between stimulus and response 
                corrs = np.array([stats.spearmanr(posdata[i,:],ausn[i,:,nAU])[0] for i in range(ntrials)])

                #Find response-stimulus lag and phase locking value using Hilbert transform
                from scipy.signal import hilbert
                posphases = np.angle(hilbert(posdatas-0.5,axis=1)) 
                ausphases = np.angle(hilbert(ausns[:,:,nAU]-0.5,axis=1))
                posphasesu = np.unwrap(posphases)
                ausphasesu = np.unwrap(ausphases)
                diffs = ausphases - posphases #phase difference between response and stimulus
                diffs[diffs > np.pi] -= 2*np.pi #a -= 360 if a > 180
                diffs[diffs < -np.pi] += 2*np.pi #a += 360 if a < -180
                lag_hilbert = np.median(diffs,axis=1) #median phase difference during each trial
                plvs = np.array([asinus_utils.PLV(posphases[trial,:],ausphases[trial,:]) for trial in range(ntrials)]) #phase locking value for each trial

                #Find response-stimulus lag using FFT, by comparing the phase of the stimulus and response at the peak frequency of the stimulus
                stim = posdata[0,:]
                freqs = np.fft.fftfreq(len(stim),d=time_interval)
                stim_fft_abs = np.abs(np.fft.fft(stim))
                peak_freq_index = np.argmax(stim_fft_abs[1:]) + 1 #index of peak frequency in stimulus
                peak_freq = freqs[1:][peak_freq_index]
                lag_fourier = np.array([asinus_utils.get_phase_lag_FFT(ausn[i,:,nAU],stim)[peak_freq_index] for i in range(ntrials)]) #phase lag in radians for each trial

                #This code section finds power in each frequency band
                #psb = np.array([acommonfuncs.power_in_band(ausn[i,:,nAU],freqs,2,4) for i in range(ntrials)])


                #Combine outcomes across similar trials
                blinks_initial = np.median(blinks[trials_bool['initial']])
                blinks_final = np.median(blinks[trials_bool['final']])
                amp_initial = np.median(amp[trials_bool['initial']])
                amp_final = np.median(amp[trials_bool['final']])
                corrs_initial = np.median(corrs[trials_bool['initial']])
                corrs_jump = np.median(corrs[trials_bool['jump']])
                corrs_final = np.median(corrs[trials_bool['final']])
                plv_initial = np.median(plvs[trials_bool['initial']])
                plv_jump = np.median(plvs[trials_bool['jump']])
                plv_final = np.median(plvs[trials_bool['final']])
                lag_hilbert_initial = np.median(lag_hilbert[trials_bool['initial']]) #note this is a median (across trials) of medians (within trials)
                lag_hilbert_final = np.median(lag_hilbert[trials_bool['final']])
                lag_fourier_initial = np.median(lag_fourier[trials_bool['initial']])
                lag_fourier_final = np.median(lag_fourier[trials_bool['final']])

                t.at[t_index,'sinus_blinks_initial'] = blinks_initial
                t.at[t_index,'sinus_blinks_final'] = blinks_final
                t.at[t_index,'sinus_amp_initial'] = amp_initial
                t.at[t_index,'sinus_amp_final'] = amp_final
                t.at[t_index,'sinus_rapidity_initial'] = rapidity_initial
                t.at[t_index,'sinus_rapidity_final'] = rapidity_final
                t.at[t_index,'sinus_corrs_initial'] = corrs_initial
                t.at[t_index,'sinus_corrs_jump'] = corrs_jump
                t.at[t_index,'sinus_corrs_final'] = corrs_final
                t.at[t_index,'sinus_plv_initial'] = plv_initial
                t.at[t_index,'sinus_plv_jump'] = plv_jump
                t.at[t_index,'sinus_plv_final'] = plv_final
                t.at[t_index,'sinus_lag_hilbert_initial'] = lag_hilbert_initial
                t.at[t_index,'sinus_lag_hilbert_final'] = lag_hilbert_final
                t.at[t_index,'sinus_lag_fourier_initial'] = lag_fourier_initial
                t.at[t_index,'sinus_lag_fourier_final'] = lag_fourier_final

                #Plot subject's data with smoothing and time points for grad-peaks
                """
                ausn_smoo = asinus_utils.lowpass(ausn[:,:,nAU],cutoff)
                ausn_smoo_grad = np.gradient(ausn_smoo,axis=1)
                ausn_smoo_grad_peaks = [signal.find_peaks(ausn_smoo_grad[i,:], distance=30)[0] for i in range(ausn_smoo_grad.shape[0])]
                fig,axs=plt.subplots(5,2,figsize=(12,8))
                for trial in range(ntrials):
                    ax = axs[np.unravel_index(trial,(5,2))]
                    ax.plot(times_trial_regular,ausn[trial,:,nAU],color='b',label='response')
                    ax.plot(times_trial_regular,ausn_smoo[trial,:],color='b',label='response',alpha=0.5)
                    jump_times = ausn_smoo_grad_peaks[trial]
                    if len(jump_times)>0:
                        for time in jump_times:
                            ax.axvline(x=times_trial_regular[time],color='r',alpha=0.5)
                    ax.set_title(f'Trial {trial+1}, {len(jump_times)}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(AU_to_plot)
                fig.suptitle(f'Subject {subject}, {ausns_grad_peaks_med_initial:.3f}')
                plt.show()
                """


                """
                #Plot subject's data
                fig,axs=plt.subplots(5,2,figsize=(12,8))
                for trial in range(ntrials):
                    ax = axs[np.unravel_index(trial,(5,2))]
                    ax.plot(times_trial_regular,ausn[trial,:,nAU],color='lightblue',label='response',alpha=0.5)
                    ax.plot(times_trial_regular,ausns[trial,:,nAU],color='darkblue',label='response',alpha=0.5)
                    ax.plot(times_trial_regular,posdata[trial,:], color='gray',label='stimulus',alpha=0.5)
                    ax.plot(times_trial_regular,posdatas[trial,:], color='black',label='stimulus',alpha=0.5)
                    jump_times = np.where(metadata[trial,:]==1)[0]
                    if len(jump_times)>0:
                        for time in jump_times:
                            ax.axvline(x=times_trial_regular[time],color='r',alpha=0.5)
                    ax.set_title(f'Trial {trial+1}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(AU_to_plot)
                fig.suptitle(f'Subject {subject}')
                """

                """
                #Plot phase relationships with Hilbert transform
                fig,axs=plt.subplots(5,2,figsize=(12,8))
                for trial in range(ntrials):
                    ax = axs[np.unravel_index(trial,(5,2))]   
                    posphase = posphases[trial,:]
                    indices = np.where(np.diff(np.sign(posphase)) == 2)[0] #indices where posphase goes from negative to positive (peaks of stimulus AU intensity)            
                    med_diff_at_peaks = np.median(diffs[trial,indices]) #median difference (radians) between stimulus and response phase at stimulus peaks
                    ax.plot(times_trial_regular,posphases[trial,:],color='k',label='stimulus',alpha=0.3)
                    ax.plot(times_trial_regular,ausphases[trial,:],color='b',label='response',alpha=0.5)
                    ax.plot(times_trial_regular,diffs[trial,:],color='r',label='diff',alpha=0.3)   
                    for index in indices:
                        ax.axvline(x=times_trial_regular[index],color='green',alpha=0.3)
                    ax.axhline(y=0,color='black',alpha=0.3)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Phase')
                    ax.set_title(f"{trial+1}. PLV {plvs[trial]:.2f}, diff {phasediff[trial]:.2f}, diff@peak {med_diff_at_peaks:.2f}")
  
                plt.show()
                assert(0)      
                """

        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_sinus.csv')

    """
    blinks, amp, rapidity, lag_hilbert, lag_fourier (for these columns we have versions for trial types 'initial' and 'final')
    corrs, plv (for these columns we have versions for all 3 trial types) 
    """

    gps = list(t[group].unique())
    if '' in gps: gps.remove('')
    hue = group
    t2 = t.loc[t.use_sinus & (t[group]!='') & ~t.sinus_outliers,:]
    sns.set_context('talk',font_scale=0.6)

    from pandas.errors import SettingWithCopyWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SettingWithCopyWarning)
        t2['sinus_lag_hilbert_abs_initial'] = np.abs(t2['sinus_lag_hilbert_initial']).values
        t2['sinus_lag_hilbert_abs_final'] = np.abs(t2['sinus_lag_hilbert_final']).values

    
    #Are there group differences in |lag| after accounting for rapidity? First use linear regression to predict lag from rapidity, then we ask whether the absolute value of residuals are different between groups
    model = smf.ols(f'sinus_lag_hilbert_initial ~ sinus_rapidity_initial', data=t2).fit()
    #get model coefficients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SettingWithCopyWarning)
        t2['sinus_lag_hilbert_res_abs_initial'] = np.abs(model.resid).values

    #Are there group differences in PLV in jump trials after accounting for PLV in initial trials?
    model = smf.ols(f'sinus_plv_jump ~ sinus_plv_initial', data=t2).fit()
    #get model coefficients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SettingWithCopyWarning)
        t2['sinus_plv_jump_res'] = np.abs(model.resid).values

    #Are there group differences in PLV after accounting for amplitude?
    print(f'ANOVA: sinus_plv_initial ~ {group} + sinus_max_val_initial') 
    print(sm.stats.anova_lm(smf.ols(f'sinus_plv_initial ~ {group} + sinus_amp_initial', data=t2).fit(), typ=2)) 

    #Are there group differences in PLV in jump trials after accounting for PLV in initial trials and amplitude?
    print(f'ANOVA: sinus_plv_jump ~ {group} + sinus_plv_initial + sinus_max_val_initial') 
    print(sm.stats.anova_lm(smf.ols(f'sinus_plv_jump ~ {group} + sinus_plv_initial + sinus_amp_initial', data=t2).fit(), typ=2)) 


    #compare same outcome in initial and final trials
    for variable in ['amp','rapidity','corrs','plv','lag_hilbert_abs']:
        for gp in gps:
            x=t2.loc[t2[group]==gp,f'sinus_{variable}_initial']
            y=t2.loc[t2[group]==gp,f'sinus_{variable}_final']
            print(f'{variable} {gp} diff {y.mean()-x.mean():.2}, p={stats.ttest_rel(x,y).pvalue:.2f}')

    #Compare outcome measures across groups
    fig,axs=plt.subplots(4,5,figsize=(12,8))
    outcomes = new_columns[2:] + ['sinus_lag_hilbert_abs_initial','sinus_lag_hilbert_abs_final','sinus_lag_hilbert_res_abs_initial','sinus_plv_jump_res']
    for i in range(len(outcomes)):
        ax = axs[np.unravel_index(i,(4,5))]
        sns.stripplot(ax=ax,data = t2, x=group, hue=hue,palette=colors,y=outcomes[i],legend=False)
        group1 = t2.loc[t2[group]==gps[0],outcomes[i]]
        group2 = t2.loc[t2[group]==gps[1],outcomes[i]]
        diff = group1.mean() - group2.mean()
        pval = stats.ttest_ind(group1,group2).pvalue
        ax.set_title(f'diff {diff:.2} p={pval:.2f}')
        ax.set_ylabel(ax.get_ylabel()[6:])
    fig.tight_layout()

    #Scatter plots for initial vs jump vs final trial type
    """
    grid=acommonfuncs.pairplot(t2,vars=['sinus_blinks_initial','sinus_blinks_final'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_amp_initial','sinus_amp_final'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_rapidity_initial','sinus_rapidity_final'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_corrs_initial','sinus_corrs_jump','sinus_corrs_final'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_plv_initial','sinus_plv_jump','sinus_plv_final'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_lag_hilbert_initial','sinus_lag_hilbert_final'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_lag_hilbert_initial_abs','sinus_lag_hilbert_final_abs'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_lag_fourier_initial','sinus_lag_fourier_final'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    """

    #Scatter plots comparing different measures
    """
    grid=acommonfuncs.pairplot(t2,vars=['sinus_plv_initial','sinus_corrs_initial'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
    grid=acommonfuncs.pairplot(t2,vars=['sinus_lag_hilbert_initial','sinus_lag_fourier_initial'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)

    grid=acommonfuncs.pairplot(t2,vars=['sinus_blinks_initial','sinus_amp_initial','sinus_rapidity_initial','sinus_corrs_initial','sinus_plv_initial','sinus_lag_hilbert_initial','sinus_lag_hilbert_abs_initial'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)

    grid=acommonfuncs.pairplot(t2,x_vars=['sinus_amp_initial','sinus_rapidity_initial','sinus_corrs_initial','sinus_plv_initial','sinus_lag_hilbert_initial','sinus_lag_hilbert_abs_initial'],y_vars=['fsiq2','panss_N','sofas'],height=1.5,kind='reg',robust=True,group=group)
    """

 



    plt.show(block=False)