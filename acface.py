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
Many trials have no response. No sudden uptick. So average time series may not be accurate

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

this_AU='AU12'
this_AU_index = aus_labels.index(this_AU)


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

    """PCA of action unit time series between trigger and stimStop when participants were asked to smile (HA)"""

    def pca_transform(pca,values):
        result=pca.transform(values)
        result=pd.DataFrame(result)
        result.columns = [f'comp{i}' for i in range(result.shape[1])]
        return result

    pca,comp0,aus_pca,aust_pca={},{},{},{}
    for emot in emots:
        pca[emot] = acface_utils.get_pca(aus_trigger2stimStop[emot])
        comp0[emot] = pca[emot].components_[0]
        aus_pca[emot] = pca_transform(pca[emot],aus)
        aust_pca[emot] = pca_transform(pca[emot],aust)
    #interp_aus_pca = interp1d(times_eachframe,aus_pca['ha'],axis=0,kind='linear',fill_value = 'extrapolate')

    """
    Get mean time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit, and for first principal component. Could use these as fMRI regressor
    In detail: Get timestamps and AU values for each trial, from trigger to next Instruct. Set timestamps to be relative to the trigger as 0. Concatenate all trials together. Find a linear interpolation of values, and resample at 20 fps to find mean time series.
    """
    relevant_timegaps=[0,0.5,0.75,1.5,0.75,0.5] #(0, trigger, post_trigger, stimMove, fixation, next instruction)
    relevant_labels=['trigger','post_trigger','stimMove','fixation','next_instruct']
    relevant_timestamps = np.cumsum(relevant_timegaps)
    midpoint_timestamps = acface_utils.calculate_averages(relevant_timestamps)   
    times_trial_regular = np.arange(0,relevant_timestamps[-1],1/target_fps) #new timestamps for resampling

    aus_trial,comp0_trial,values_resampled,aus_trial_mean={},{},{},{}
    """
    Each variable above is a dictionary with keys 'ha' and 'an', for when pt was asked to smile or frown
    aus_trial['ha'] is n_smiletrials (40) * n_frames (80) * nAUs (16)
    """
    for emot in emots:
        aus_trial[emot] = acface_utils.get_all_post_trigger_time_series(df,interp_aus,times_trial_regular,emotion=emot)
        comp0_trial[emot] = np.vstack([pca[emot].transform(i)[:,0] for i in aus_trial[emot]]) #n_smiletrials * n_frames
        aus_trial_mean[emot]=np.mean(aus_trial[emot],axis=0) #mean across trials

        values_resampled[emot] = acface_utils.get_mean_post_trigger_time_series(times_trial_regular,aus_trial[emot])

    """Outcome 1: Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12))"""
    aus_trial_ha_max = np.max(aus_trial['ha'],axis=1)
    aus_trial_ha_min = np.min(aus_trial['ha'],axis=1)
    aus_trial_ha_range = aus_trial_ha_max - aus_trial_ha_min
    this_au_trial_ha_max = aus_trial_ha_max[:,this_AU_index] #RETURN
    this_au_trial_ha_range = aus_trial_ha_range[:,this_AU_index] #RETURN



    fig,ax=plt.subplots()
    ax.plot(this_au_trigger2stimStop_HA_max,color='red')
    ax.plot(this_au_trial_ha_max,color='blue')
    ax.set_title('max')

    fig,ax=plt.subplots()
    ax.plot(this_au_trigger2stimStop_HA_range,color='red')
    ax.plot(this_au_trial_ha_range,color='blue')
    ax.set_title('range')





    emot='ha'
    values = aus_trial[emot][:,:,this_AU_index]
    #values = comp0_trial[emot]

    fig,axs=plt.subplots(nrows=4,ncols=4)
    fig.set_size_inches(18,8)
    for i in range(16): #plot examples of time series from a few trials
        ax = axs[np.unravel_index(i,(4,4))]
        ax.plot(times_trial_regular,values[i,:],color='blue')
        if np.min(values) >= 0:
            ax.set_ylim(0,3)
        for j in relevant_timestamps:
            ax.axvline(x=j) 
        for i,annotation in enumerate(relevant_labels):
            ax.text(midpoint_timestamps[i],+0.5,annotation,ha='center')
    fig.tight_layout()

    fig,ax=plt.subplots()
    for i in range(aus_trial[emot].shape[0]):
        ax.plot(times_trial_regular,aus_trial[emot][i,:,this_AU_index],color='blue',linewidth=0.2)
    ax.plot(times_trial_regular,values_resampled[emot][:,this_AU_index],color='r',linewidth=1) #plot mean time series
    ax.plot(times_trial_regular,aus_trial_mean[emot][:,this_AU_index],color='k',linewidth=1) #plot mean time series
    #ax.set_ylim(0,3)
    for j in relevant_timestamps:
        ax.axvline(x=j) 
    for i,annotation in enumerate(relevant_labels):
        ax.text(midpoint_timestamps[i],-0.5,annotation,ha='center')
    fig.tight_layout()





    
    fig,ax=plt.subplots()
    fig.set_size_inches(18,3) #40,5
    acface_utils.plot_this_au(df,ax,times_regular,aust,this_AU='AU12')
    acface_utils.plot_all_aus(df,times_regular,aust)
    fig,ax=plt.subplots()
    fig.set_size_inches(18,3) #40,5
    acface_utils.plot_this_au(df,ax,times_regular,aust_pca['ha'],this_AU='comp0',color='blue')
    acface_utils.plot_this_au(df,ax,times_regular,aust_pca['an'],this_AU='comp0',color='red')
    #acface_utils.plot_all_aus(df,times_regular,aust_pca['ha'])   
    #acface_utils.plot_all_aus(df,times_regular,aust_pca['an'])
    #print([[aus_names[i],comp0[i]] for i in range(len(comp0))] )
    
    plt.show()
    assert(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot]))

    #Plot to see whether response amplitudes are decreasing with time due to tiredness
    """
    plt.scatter(range(len(this_au_trigger2stimStop_HA_max)) , this_au_trigger2stimStop_HA_max) 
    plt.xlabel('Trial')
    plt.ylabel('Max AU12 value from trigger to stimStop')
    """

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
