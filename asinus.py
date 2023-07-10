"""
Analyse Facial mirring task (sinus)
10 trials, each lasting 20s

Webcam recording received at 30fps, but task saved data (and stimulus display) are at 25 fps (0.04 sec/frame)

First 2 trials are no feedback, predictable
Next 5 trials are no feedback, pjump=[0.025,0.03,0.035,0.04,0.045]
Last 3 trials are feedback present, predictable

New columns
    use_sinus
    sinus_outliers
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
import re, scipy.io
import warnings
from acommonvars import *
import acommonfuncs


def get_sinus_data(subject):
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\sinus*\\") #find the FF1 folder for this subject
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

nseconds_per_trial = nframes*time_interval
times_trial_regular = np.linspace(time_interval, nseconds_per_trial, nframes)

outliers=[] #exclude these subjects


_,posdata,metadata = get_sinus_data('020') #posdata and metadata are same for every subject

stimulus = pd.read_csv(f'{analysis_folder}\\sinus\\F01-NE-HA_OpenFace_static.csv')[f' {AU_to_plot}_r']

posdata=np.vectorize(lambda x: stimulus[x])(posdata) #pos data contains which frame number of stimulus face (0 to 30) was shown at each time point. Convert these to stimulus AU intensity values

if __name__=='__main__':
    c = acommonfuncs.clock()

    ### SETTABLE PARAMETERS 
    group = 'group02' #the grouping variable
    load_table=False

    ### Get data
    new_columns = ['use_sinus','sinus_outliers','sinus_ts']

    if load_table:
        t = acommonfuncs.add_table(t,'outcomes_sinus.csv')
        t = acommonfuncs.str_columns_to_literals(t,['sinus_ts'])
    else:
        t['use_sinus'] = ((include) & (t.valid_sinuso==1)) 
        t['prop_outliers'] = t.subject.isin(outliers)
        t=acommonfuncs.add_columns(t,['sinus_ts'])
        for t_index in range(len(t)):
            if t['use_sinus'][t_index]:
                subject=t.subject[t_index]
                print(f'{c.time()[1]}: Subject {subject}')


                ausdata,_,_ = get_sinus_data(subject)


                #ausdata[7,3], ausdata[0,3]
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
                blinks = [np.sum(aus[trial,-1,:]>0)/nframes for trial in range(ntrials)] #proportion of blinks
                max_vals = [np.max(aus[trial,:,nAU]) for trial in range(ntrials)] #max intensity of AU12

                #Normalize each trial's time series to range (0,1)
                from sklearn import preprocessing 
                posdata = preprocessing.minmax_scale(posdata, feature_range=(0, 1), axis=1, copy=True)
                aus2 = np.transpose(aus,axes=[0,2,1])
                aus3 = preprocessing.minmax_scale(aus2.reshape(-1,aus2.shape[-1]), feature_range=(0, 1), axis=1, copy=True).reshape(aus2.shape)
                ausn = np.transpose(aus3, axes=[0,2,1])


                #looping over i will be messed up by little loops within i... do my other codes have this problem?
                #or try wtc. (wavelet coherence). Get phase lag, and coherence value


                def lowpass_filter(data,lowcut,fs,axis=-1):
                    #try lowcut = 1, 3, 4
                    from scipy.signal import butter, lfilter
                    order = 10
                    b, a = butter(order, lowcut, btype='low', fs=fs, output='ba')
                    y = lfilter(b, a, data,axis=axis)
                    return y                
                
                #lowcut = 1 #low pass cutoff in Hz
                #ausn = lowpass_filter(ausn,lowcut,int(1/time_interval),axis=1)

                
                #Plot subject's data
                fig,axs=plt.subplots(5,2,figsize=(12,8))
                for trial in range(ntrials):
                    ax = axs[np.unravel_index(trial,(5,2))]
                    ax.plot(times_trial_regular,ausn[trial,:,nAU],color='b',label='response')
                    ax.plot(times_trial_regular,posdata[trial,:], color='k',label='stimulus',alpha=0.5)
                    jump_times = np.where(metadata[trial,:]==1)[0]
                    if len(jump_times)>0:
                        for time in jump_times:
                            ax.axvline(x=times_trial_regular[time],color='r',alpha=0.5)
                    ax.set_title(f'Trial {trial+1}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(AU_to_plot)
                

                def PLV(x,y):
                    #calculate phase locking value of vectors x and y
                    e = np.exp(1j * (x - y))
                    return np.abs(np.sum(e)) / len(e)
                
                fig,axs=plt.subplots(5,2,figsize=(12,8))
                for trial in range(ntrials):
                    ax = axs[np.unravel_index(trial,(5,2))]               
                    thisposdata = posdata[trial,:]
                    thisausdata = ausn[trial,:,nAU]

                    from scipy.signal import hilbert
                    posphase = np.angle(hilbert(thisposdata-0.5))
                    auphase = np.angle(hilbert(thisausdata-0.5))
                    posphaseu = np.unwrap(posphase)
                    auphaseu = np.unwrap(auphase)
                    
                    plv = PLV(posphaseu,auphaseu)

                    indices = np.where(np.diff(np.sign(posphase)) == 2)[0] #indices where posphase goes from negative to positive (peaks of stimulus AU intensity)
                    diffs = auphase - posphase
                    diffs[diffs > np.pi] -= 2*np.pi #a -= 360 if a > 180
                    diffs[diffs < -np.pi] += 2*np.pi #a += 360 if a < -180

                    #find peaks in diffs using scipy findpeaks. THen look at the gradient of diffs. This is basically how the rate of change of phase is different between response and stimulus. The median of these differences indicates the rapidity of the person's facial response. Might need to smooth data for this to work well.
                    from scipy.signal import find_peaks
                    diffs_grad = np.gradient(diffs)
                    peaks, _ = find_peaks(diffs_grad, height=0)
                    median_diffgrad = np.median(diffs_grad[peaks]) #median (among local parks) of gradient of diffs      
                

                    mean_diff_at_peaks = np.mean(diffs[indices]) #mean difference (radians) between stimulus and response phase at stimulus peaks
                    mean_diff = np.mean(diffs)

                    #print(f'subject {subject}, mean_diff_at_peaks {mean_diff_at_peaks:.2f}, mean_diff {mean_diff:.2f}')

                    ax.plot(times_trial_regular,posphase,color='k',label='stimulus',alpha=0.3)
                    ax.plot(times_trial_regular,auphase,color='b',label='response',alpha=0.5)
                    ax.plot(times_trial_regular,diffs,color='r',label='diff',alpha=0.3)   
                    for index in indices:
                        ax.axvline(x=times_trial_regular[index],color='green',alpha=0.3)
                    ax.axhline(y=0,color='black',alpha=0.3)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Phase')
                    ax.set_title(f"{trial+1}. PLV {plv:.2f}, diff {mean_diff:.2f}, grad {median_diffgrad:.2f}")
                #fig.tight_layout()
                plt.show()
                #assert(0)      

                """
                sinus_ts=get_sinus_data(subject) 
                t.at[t_index,'prop_stim'] = list(stimface)
                t.at[t_index,'prop_resp'] = list(respface)

                t.at[t_index,'prop_stim_min'] = min(stimface)
                t.at[t_index,'prop_stim_max'] = max(stimface)
                t.at[t_index,'prop_stim_range'] = t.prop_stim_max[t_index] - t.prop_stim_min[t_index]

                if t.subject[t_index] not in outliers:
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stimface,respface)
                    t.at[t_index,'prop_slope'] = slope
                    t.at[t_index,'prop_intercept'] = intercept
                    t.at[t_index,'prop_r2'] = r_value**2
                """

        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_sinus.csv')
