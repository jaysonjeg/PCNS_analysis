"""
Analyse cface1 face data
Mainly using the out.csv log file, and the OpenFace-processed .csv file with action unit time series
Within each trial, key events are 'trigger', 'stimMove', 'stimStop', 'fixation'
Resample action unit series from being indexed by frames, to be indexed by time (sec)

Get the following metrics of facial movement for each subject:
1) amplitudes: Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12)). Does this decrease with time due to tiredness?
2) mean_timeseries: Mean time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit. Could use these as fMRI regressor
3) mean_timeseries_pca: Point 3 but for first principal component
4) latencies: Distribution of latency post-trigger for the smile or frown action. Look at distribution and exclude outliers
5) maxgrad: Maximum value of first derivative for AU12 or PCA component

Deal with amplitude as confounder

Issues:
- Many trials have no response. No sudden uptick. So average time series may not be accurate
- Even trials with a response are not well modelled by average time series, which is a smooth curve. This is because the average time series is a smooth curve, whereas the actual time series is more like a step function.

"""


import numpy as np, pandas as pd, matplotlib.pyplot as plt
import acommonfuncs, acface_utils
from acommonvars import *
from scipy.interpolate import interp1d
c = acommonfuncs.clock()

### SETTABLE PARAMETERS ###
target_fps=20
ntrials=80
n_trialsperemotion=40
emots=['ha','an']
static_or_dynamic = 'static' #whether au_static was used in OpenFace execution or not
action_unit = 'AU12' #which action unit to plot
min_success = 0.90 #minimum proportion of successful frames for a subject to be included

ha_AU='AU12'
ha_AU_index = aus_labels.index(ha_AU)

get_amplitudes=True
get_mean_timeseries=False
get_mean_timeseries_pca=False 
get_latencies=False 
get_maxgrads=False
to_plot=False

HC=((healthy_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #healthy group
PT=((clinical_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #patient group
SZ = ((sz_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #schizophrenia subgroup
SZA = ((sza_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #schizoaffective subgroup
HC,PT,SZ,SZA = subs[HC],subs[PT],subs[SZ],subs[SZA]

def get_metrics(subject):
    print(c.time()[1])
    all_frames,aus,success = acommonfuncs.get_openface_table('cface1',subject,static_or_dynamic) #Get the OpenFace intermediates .csv for this subject

    webcam_frames_success_proportion = np.sum(success)/len(success)
    if webcam_frames_success_proportion < min_success:
        print(f"WARNING: {subject} has only {webcam_frames_success_proportion:.3f} proportion of successful frames. Returning nan.")
        return np.nan

    df = acommonfuncs.get_beh_data('cface1',subject,'out',use_MRI_task=False) #Get behavioural data from *out.csv

    """Get face summary data from *face.csv"""
    face_summary = acommonfuncs.get_beh_data('cface1',subject,'face',use_MRI_task=False)
    #face_summary=pd.read_csv(glob(f"{resultsFolder}*face.csv")[0]) # make face summary csv into dataframe 
    face_summary = {i[0]:i[1] for i in face_summary.values} #convert face summary array into dictionary
    camtstart = face_summary['camtstart'] #time when webcam started recording
    camactualfps = face_summary['camactualfps'] #actual fps of webcam recording

    """
    From out.csv log file, get all pairs of (time,framenumber). Interpolate between frames to calculate a time since onset for each frame number in webcam data. Now the action unit time series is indexed by timestamp instead of frame number. Then interpolate timestamps to resample the AU time series at regular intervals of exactly 20fps.
    """
    times,frames = acface_utils.get_all_timestamps_and_framestamps(df,ntrials)
    duplicate_indices = acface_utils.find_duplicate_indices(times) + acface_utils.find_duplicate_indices(frames)
    valid_indices = [i for i in range(len(frames)) if i not in duplicate_indices] #remove entries where same frame number corresponds to multiple timestamps, or timestamp corresponds to multiple frame numbers (usually due to dropped frames)
    interp_frametimes = interp1d(frames[valid_indices],times[valid_indices],kind='linear',fill_value = 'extrapolate')
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
    relevant_timejitters=[0,0.5,0.75,1.5,0.75,0.5] #(0, trigger, post_trigger, stimMove, fixation, next instruction)
    relevant_labels=['trigger','post_trigger','stimMove','fixation','next_instruct']
    relevant_timestamps = np.cumsum(relevant_timejitters)
    midpoint_timestamps = acface_utils.calculate_averages(relevant_timestamps)   
    times_trial_regular = np.arange(0,relevant_timestamps[-1],1/target_fps) #new timestamps for resampling

    """
    Get interpolated AU time series for each trial separately.
    Each variable below is a dictionary with keys 'ha' and 'an', for when pt was asked to smile or frown
    aus_trial['ha'] has interpolated AU values for each trial separately: array size n_smiletrials (40) * n_frames (80) * nAUs (16)
    """
    aus_trial={}
    for emot in emots:
        aus_trial[emot] = acface_utils.get_all_post_trigger_time_series(df,interp_aus,times_trial_regular,emotion=emot)

    if get_amplitudes:
        """Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12))"""
        aus_trial_max,aus_trial_min,aus_trial_range = {},{},{}
        for emot in emots:
            aus_trial_max[emot] = np.max(aus_trial[emot],axis=1)
            aus_trial_min[emot] = np.min(aus_trial[emot],axis=1)
            aus_trial_range[emot] = aus_trial_max[emot] - aus_trial_min[emot]
        ha_AU_trial_ha_max = aus_trial_max['ha'][:,ha_AU_index] #OUTCOME amplitudes
        ha_AU_trial_ha_range = aus_trial_range['ha'][:,ha_AU_index] #OUTCOME amplitudes

    if get_mean_timeseries:
        aus_trial_mean2,aus_trial_mean={},{} #aus_trial_mean['ha'] has mean across trials, separately for each timepoint
        for emot in emots:
            aus_trial_mean[emot]=np.mean(aus_trial[emot],axis=0) #OUTCOME mean_timeseries
            aus_trial_mean2[emot] = acface_utils.get_mean_post_trigger_time_series(times_trial_regular,aus_trial[emot]) #alternative method (not good)

    if get_mean_timeseries_pca:
        #PCA of action unit time series for each emotion separately
        pca,comp0,aus_pca,aus_trial_pca,aust_pca,aus_trial_pca_mean,aus_trial_pca_mean2={},{},{},{},{},{},{}
        for emot in emots:
            pca[emot] = acface_utils.get_pca(aus_trial[emot])
            if not(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot])):
                pca[emot].components_[0] = -pca[emot].components_[0] #ensure component zero increases from trigger to middle of stimMove
            comp0[emot] = pca[emot].components_[0]
            aus_trial_pca[emot] = np.array([pca[emot].transform(i) for i in aus_trial[emot]]) 
            aus_trial_pca_mean[emot] = pca[emot].transform(aus_trial_mean[emot]) #OUTCOME mean_timeseries_pca
            aus_trial_pca_mean2[emot] = pca[emot].transform(aus_trial_mean2[emot])
            aus_pca[emot] = acface_utils.pca_transform(pca[emot],aus) 
            aust_pca[emot] = acface_utils.pca_transform(pca[emot],aust) 
        #interp_aus_pca = interp1d(times_eachframe,aus_pca['ha'],axis=0,kind='linear',fill_value = 'extrapolate')  
        assert(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot]))

    if get_latencies and get_maxgrads:
        get_latency = lambda values: acface_utils.get_latency(values,target_fps,n_trialsperemotion,times_trial_regular,plot=False)
        r_an_pca0 = get_latency(aus_trial_pca['an'][:,:,0])
        r_ha_pca0 = get_latency(aus_trial_pca['ha'][:,:,0])
        r_ha_AU12 = get_latency(aus_trial_pca['ha'][:,:,ha_AU_index])
        #acface_utils.plot_this_au_trial(r_an_pca0,'sub sz 0 - an - comp0',times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps,plot_relevant_timestamps=False,results=results)

        r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(r_ha_pca0,n_trialsperemotion)    
        r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(r_an_pca0,n_trialsperemotion)
        #r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(r_ha_AU12,n_trialsperemotion) 

    """Plotting"""
    if to_plot:       
        plot_this_au_trial = lambda values,title: acface_utils.plot_this_au_trial(values,title,times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps)
        emot='ha'
        values = aus_trial[emot][:,:,ha_AU_index]
        title = f'{ha_AU} time series for some {emot} trials'
        plot_this_au_trial(values,title)
        emot='an'
        values = aus_trial_pca[emot][:,:,0]
        title = f'PCA comp 0 time series for some {emot} trials'
        plot_this_au_trial(values,title)

        title = f'{ha_AU} time series for all SMILE trials'
        acface_utils.plot_this_au_trial_superimposed('ha',ha_AU_index,title,aus_trial,aus_trial_mean,aus_trial_mean2,relevant_timestamps,relevant_labels,midpoint_timestamps,times_trial_regular)
        title = f'PCA comp 0 time series for all FROWN trials'
        acface_utils.plot_this_au_trial_superimposed('an',0,title,aus_trial_pca,aus_trial_pca_mean,aus_trial_pca_mean2,relevant_timestamps,relevant_labels,midpoint_timestamps,times_trial_regular)

        fig,ax=plt.subplots()
        fig.set_size_inches(18,3) #40,5
        acface_utils.plot_this_au(df,ax,times_regular,aust,this_AU='AU12',color='blue',label='smile')

        fig,ax=plt.subplots()
        fig.set_size_inches(18,3) #40,5
        acface_utils.plot_this_au(df,ax,times_regular,aust_pca['ha'],this_AU='comp0',color='blue',label='smile')
        acface_utils.plot_this_au(df,ax,times_regular,aust_pca['an'],this_AU='comp0',color='red',label='frown')

        #acface_utils.plot_all_aus(df,times_regular,aust)
        #acface_utils.plot_all_aus(df,times_regular,aust_pca['ha'])   
        #acface_utils.plot_all_aus(df,times_regular,aust_pca['an'])
        #print([[aus_names[i],comp0[i]] for i in range(len(comp0))] )

        #Plot to see whether response amplitudes are decreasing with time due to tiredness
        fig,ax = plt.subplots()
        ax.scatter(range(len(ha_AU_trial_ha_max)) , ha_AU_trial_ha_max) 
        ax.set_xlabel('Trial')
        ax.set_ylabel('Max AU12 value from trigger to end of nextInstruct')

        #acface_utils.plot_pca_mapping(pca, aus_pca,aus_names)
        plt.show()

    return ha_AU_trial_ha_max

r_HC = [get_metrics(sub) for sub in HC[:]]
r_SZ = [get_metrics(sub) for sub in SZ[:]]
r_SZA = [get_metrics(sub) for sub in SZA[:]]

r_HC_mean = [np.mean(i) for i in r_HC]
r_SZ_mean = [np.mean(i) for i in r_SZ]
r_SZA_mean = [np.mean(i) for i in r_SZA]

def plot_amplitudes_decreasing(results,title):
    fig,axs=plt.subplots(4,4)
    axs = axs.flatten()
    for i in range(16):
        axs[i].scatter(range(len(results[i])),results[i])  
        axs[i].set_xlabel('Trial')
        axs[i].set_title(title)

plot_amplitudes_decreasing(r_HC,'HC')
plt.show()