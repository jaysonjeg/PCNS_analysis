"""
Analyse cface1 face data
Mainly using the out.csv log file, and the OpenFace-processed .csv file with action unit time series
Within each trial, key events are 'trigger', 'stimMove', 'stimStop', 'fixation'
Resample action unit series from being indexed by frames, to be indexed by time (sec)

Get the following metrics of facial movement for each subject:
1) Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12)). Does this decrease with time due to tiredness?
2) Mean time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit, and for first principal component. Could use these as fMRI regressor
3) Distribution of latency post-trigger for the smile or frown action. 

Issues:
- Many trials have no response. No sudden uptick. So average time series may not be accurate
- Even trials with a response are not well modelled by average time series, which is a smooth curve. This is because the average time series is a smooth curve, whereas the actual time series is more like a step function.

"""

import numpy as np, pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from acommon import *
import acface_utils
from scipy.interpolate import interp1d

target_fps=20
ntrials=80
emots=['ha','an']
static_or_dynamic = 'static' #whether au_static was used in OpenFace execution or not
action_unit = 'AU12' #which action unit to plot
min_success = 0.95 #minimum proportion of successful frames for a subject to be included

ha_AU='AU12'
ha_AU_index = aus_labels.index(ha_AU)

HC=((healthy_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #healthy group
PT=((clinical_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #patient group
SZ = sz_didmri_inc #schizophrenia subgroup
SZA = sza_didmri_inc #schizoaffective subgroup
HC,PT,SZ,SZA = subs[HC],subs[PT],subs[SZ],subs[SZA]

for subject in HC[0:1]:
    """Get the OpenFace intermediates .csv for this subject"""
    contents = glob(f"{intermediates_folder}\\per_subject\\{subject}\\cface1\\")
    assert(len(contents)==1)
    resultsFolder=contents[0]
    face = pd.read_csv(glob(f"{resultsFolder}\\OpenFace_{static_or_dynamic}\\*_cam_20fps.csv")[0])
    all_frames=np.asarray(face['frame'])
    success=np.array(face[' success'])
    assert(np.sum(success)/len(success) > min_success) #check that most webcam frames are successful
    aus_labels_r = [f' {i}_r' for i in aus_labels] #Convert action unit labels into column labels for OpenFace .csv file
    aus_labels_c = [f' {i}_c' for i in aus_labels] 
    aus = face[aus_labels_r] #get all action units' time series for this subject. The rows are numbered from zero, whereas actual frames are numbered from 1. The error (1/20th of a second) is negligible.
    aus_c = face[aus_labels_c]
    aus.columns=aus_labels #rename columns, for example from ' AU01_r' to 'AU01'
    aus_c.columns=aus_labels #from ' AU01_c' to 'AU01'

    """Get behavioural data from *out.csv"""
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\cface1*Ta_H*\\") #'cface' task non-MRI folder for this particular subject
    assert(len(contents)==1) #make sure exactly one matching file
    resultsFolder=contents[0]
    df=pd.read_csv(glob(f"{resultsFolder}*out.csv")[0]) # make log csv into dataframe

    """Get face summary data from *face.csv"""
    face_summary=pd.read_csv(glob(f"{resultsFolder}*face.csv")[0]) # make face summary csv into dataframe 
    face_summary = {i[0]:i[1] for i in face_summary.values} #convert face summary array into dictionary
    camtstart = face_summary['camtstart'] #time when webcam started recording
    camactualfps = face_summary['camactualfps'] #actual fps of webcam recording

    """
    From out.csv log file, get all pairs of (time,framenumber). Interpolate between frames to calculate a time since onset for each frame number in webcam data. Now the action unit time series is indexed by timestamp instead of frame number. Then interpolate timestamps to resample the AU time series at regular intervals of exactly 20fps.
    """
    times,frames = acface_utils.get_all_timestamps_and_framestamps(df,ntrials)
    interp_frametimes = interp1d(frames, times,kind='linear',fill_value = 'extrapolate')
    times_eachframe = interp_frametimes(all_frames) #get time since onset for each frame number, corresponding to rows of aus
    interp_aus = interp1d(times_eachframe, aus, axis=0, kind='linear',fill_value = 'extrapolate')
    times_regular = np.arange(0,np.max(times_eachframe),1/target_fps)
    aust = interp_aus(times_regular) 
    aust = pd.DataFrame(aust)
    aust.columns=aus.columns

    """
    Get mean time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit, and for first principal component. Could use these as fMRI regressor
    In detail: Get timestamps and AU values for each trial, from trigger to next Instruct. Set timestamps to be relative to the trigger as 0. Concatenate all trials together. Find a linear interpolation of values, and resample at 20 fps to find mean time series.
    """
    relevant_timegaps=[0,0.5,0.75,1.5,0.75,0.5] #(0, trigger, post_trigger, stimMove, fixation, next instruction)
    relevant_labels=['trigger','post_trigger','stimMove','fixation','next_instruct']
    relevant_timestamps = np.cumsum(relevant_timegaps)
    midpoint_timestamps = acface_utils.calculate_averages(relevant_timestamps)   
    times_trial_regular = np.arange(0,relevant_timestamps[-1],1/target_fps) #new timestamps for resampling

    """
    Each variable below is a dictionary with keys 'ha' and 'an', for when pt was asked to smile or frown
    aus_trial['ha'] has interpolated AU values for each trial separately: array size n_smiletrials (40) * n_frames (80) * nAUs (16)
    aus_trial_mean['ha'] has mean across trials, separately for each timepoint
    """
    aus_trial,aus_trial_mean2,aus_trial_mean={},{},{}
    for emot in emots:
        aus_trial[emot] = acface_utils.get_all_post_trigger_time_series(df,interp_aus,times_trial_regular,emotion=emot)
        aus_trial_mean[emot]=np.mean(aus_trial[emot],axis=0) #OUTCOME 2
        aus_trial_mean2[emot] = acface_utils.get_mean_post_trigger_time_series(times_trial_regular,aus_trial[emot]) #alternative method (not good)

    """Outcome 1: Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12))"""
    aus_trial_max,aus_trial_min,aus_trial_range = {},{},{}
    for emot in emots:
        aus_trial_max[emot] = np.max(aus_trial[emot],axis=1)
        aus_trial_min[emot] = np.min(aus_trial[emot],axis=1)
        aus_trial_range[emot] = aus_trial_max[emot] - aus_trial_min[emot]
    ha_AU_trial_ha_max = aus_trial_max['ha'][:,ha_AU_index] #OUTCOME 1
    ha_AU_trial_ha_range = aus_trial_range['ha'][:,ha_AU_index] #OUTCOME 1

    """PCA of action unit time series for each emotion separately"""
    pca,comp0,aus_pca,aus_trial_pca,aust_pca={},{},{},{},{}
    for emot in emots:
        pca[emot] = acface_utils.get_pca(aus_trial[emot])
        if not(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot])):
            pca[emot].components_[0] = -pca[emot].components_[0] #ensure component zero increases from trigger to middle of stimMove
        comp0[emot] = pca[emot].components_[0]
        aus_trial_pca[emot] = np.array([pca[emot].transform(i) for i in aus_trial[emot]]) 
        aus_trial_pca_mean = pca[emot].transform(aus_trial_mean[emot]) #OUTCOME 2 PCA
        aus_trial_pca_mean2 = pca[emot].transform(aus_trial_mean2[emot])
        aus_pca[emot] = acface_utils.pca_transform(pca[emot],aus)
        aust_pca[emot] = acface_utils.pca_transform(pca[emot],aust)
    #interp_aus_pca = interp1d(times_eachframe,aus_pca['ha'],axis=0,kind='linear',fill_value = 'extrapolate')  

    assert(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot]))

    """Plotting"""

    emot='ha'
    values = aus_trial[emot][:,:,ha_AU_index]
    #values = aus_trial_pca[emot][:,0]
    title = f'{ha_AU} time series for some {emot} trials'
    acface_utils.plot_this_au_trial(values,title)


    #On a single plot, plot each trial's single-AU time series in blue, then mean time course in black
    emot='ha'
    title = f'{ha_AU} time series for some {emot} trials'

    def plot_this_au_trial_superimposed(index,title,aus_trial,aus_trial_mean,aus_trial_mean2,relevant_timestamps,relevant_labels,midpoint_timestamps):
        fig,ax=plt.subplots()
        for i in range(aus_trial[emot].shape[0]):
            ax.plot(times_trial_regular,aus_trial[emot][i,:,ha_AU_index],color='blue',linewidth=0.2)
        ax.plot(times_trial_regular,aus_trial_mean[emot][:,ha_AU_index],color='k',linewidth=1,label='mean') #plot mean time series
        ax.plot(times_trial_regular,aus_trial_mean2[emot][:,ha_AU_index],color='r',linewidth=1,label='mean2') #plot mean time series, alternate method
        ax.legend()
        ax.set_title(title)
        #ax.set_ylim(0,3)
        for j in relevant_timestamps:
            ax.axvline(x=j) 
        for i,annotation in enumerate(relevant_labels):
            ax.text(midpoint_timestamps[i],-0.5,annotation,ha='center')
        fig.tight_layout()


    #More plots
    acface_utils.plot_all_aus(df,times_regular,aust)

    fig,ax=plt.subplots()
    fig.set_size_inches(18,3) #40,5
    acface_utils.plot_this_au(df,ax,times_regular,aust,this_AU='AU12',color='blue',label='smile')

    fig,ax=plt.subplots()
    fig.set_size_inches(18,3) #40,5
    acface_utils.plot_this_au(df,ax,times_regular,aust_pca['ha'],this_AU='comp0',color='blue',label='smile')
    acface_utils.plot_this_au(df,ax,times_regular,aust_pca['an'],this_AU='comp0',color='red',label='frown')

    #acface_utils.plot_all_aus(df,times_regular,aust_pca['ha'])   
    #acface_utils.plot_all_aus(df,times_regular,aust_pca['an'])
    #print([[aus_names[i],comp0[i]] for i in range(len(comp0))] )
    
    plt.show()

    #Plot to see whether response amplitudes are decreasing with time due to tiredness
    
    plt.scatter(range(len(ha_AU_trial_ha_max)) , ha_AU_trial_ha_max) 
    plt.xlabel('Trial')
    plt.ylabel('Max AU12 value from trigger to end of nextInstruct')
    

    # Plot how PCA components map onto action units
    """
    fig, ax = plt.subplots()
    im = ax.imshow(pca['ha'].components_.T)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('AU')
    ax.set_yticks(np.arange(len(aus_names)))
    ax.set_yticklabels(aus_names)
    ax.set_xticks(np.arange(len(aus_pca['ha'].columns)))
    ax.set_xlabel('Component')
    ax.set_xticklabels([i[-1] for i in aus_pca['ha'].columns])
    """

    


plt.show()
