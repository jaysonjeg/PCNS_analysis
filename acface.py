"""
Analyse cface1 face data
Mainly using the out.csv log file, and the OpenFace-processed .csv file with action unit time series
Within each trial, key events are 'trigger', 'stimMove', 'stimStop', 'fixation'
Resample action unit series from being indexed by frames, to be indexed by time (sec)

Get the following metrics of facial movement for each subject:
1) Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12)). Does this decrease with time due to tiredness?
2) Distribution of latency post-trigger for the smile or frown action. Look at distribution and exclude outliers
3) Mean time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit. Could use these as fMRI regressor
4) Point 3 but for first principal component

Deal with amplitude as confounder
Max of first derivative??

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
n_trialsperemotion=40
emots=['ha','an']
static_or_dynamic = 'static' #whether au_static was used in OpenFace execution or not
action_unit = 'AU12' #which action unit to plot
min_success = 0.95 #minimum proportion of successful frames for a subject to be included

ha_AU='AU12'
ha_AU_index = aus_labels.index(ha_AU)

HC=((healthy_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #healthy group
PT=((clinical_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #patient group
SZ = ((sz_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #schizophrenia subgroup
SZA = ((sza_didmri_inc) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #schizoaffective subgroup
HC,PT,SZ,SZA = subs[HC],subs[PT],subs[SZ],subs[SZA]

for subject in SZ[1:2]:
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
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\cface1*Ta_*\\") #'cface' task non-MRI folder for this particular subject
    contents = [i for i in contents if 'Ta_M' not in i]

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
    Outcome 3:
    Each variable below is a dictionary with keys 'ha' and 'an', for when pt was asked to smile or frown
    aus_trial['ha'] has interpolated AU values for each trial separately: array size n_smiletrials (40) * n_frames (80) * nAUs (16)
    aus_trial_mean['ha'] has mean across trials, separately for each timepoint
    """
    aus_trial,aus_trial_mean2,aus_trial_mean={},{},{}
    for emot in emots:
        aus_trial[emot] = acface_utils.get_all_post_trigger_time_series(df,interp_aus,times_trial_regular,emotion=emot)
        aus_trial_mean[emot]=np.mean(aus_trial[emot],axis=0) #OUTCOME 3
        aus_trial_mean2[emot] = acface_utils.get_mean_post_trigger_time_series(times_trial_regular,aus_trial[emot]) #alternative method (not good)

    """Outcome 1: Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12))"""
    aus_trial_max,aus_trial_min,aus_trial_range = {},{},{}
    for emot in emots:
        aus_trial_max[emot] = np.max(aus_trial[emot],axis=1)
        aus_trial_min[emot] = np.min(aus_trial[emot],axis=1)
        aus_trial_range[emot] = aus_trial_max[emot] - aus_trial_min[emot]
    ha_AU_trial_ha_max = aus_trial_max['ha'][:,ha_AU_index] #OUTCOME 1
    ha_AU_trial_ha_range = aus_trial_range['ha'][:,ha_AU_index] #OUTCOME 1

    """Outcome 4: PCA of action unit time series for each emotion separately"""
    pca,comp0,aus_pca,aus_trial_pca,aust_pca,aus_trial_pca_mean,aus_trial_pca_mean2={},{},{},{},{},{},{}
    for emot in emots:
        pca[emot] = acface_utils.get_pca(aus_trial[emot])
        if not(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot])):
            pca[emot].components_[0] = -pca[emot].components_[0] #ensure component zero increases from trigger to middle of stimMove
        comp0[emot] = pca[emot].components_[0]
        aus_trial_pca[emot] = np.array([pca[emot].transform(i) for i in aus_trial[emot]]) 
        aus_trial_pca_mean[emot] = pca[emot].transform(aus_trial_mean[emot]) #OUTCOME 4
        aus_trial_pca_mean2[emot] = pca[emot].transform(aus_trial_mean2[emot])
        aus_pca[emot] = acface_utils.pca_transform(pca[emot],aus) 
        aust_pca[emot] = acface_utils.pca_transform(pca[emot],aust) 
    #interp_aus_pca = interp1d(times_eachframe,aus_pca['ha'],axis=0,kind='linear',fill_value = 'extrapolate')  

    """Outcome 2: Latency"""

    au_index = 0 #0 for pca or ha_AU_index for AU12
    values = aus_trial_pca['an'][:,:,au_index] #n_trialsperemotion (40) * n_frames (80)

    #Timepoints to compare to make sure valid facial response was detected
    default_indices_lower = [5] #250ms after trigger (optionally add at start of nextInstruct to make sure they look normal again)
    default_indices_upper = [int(target_fps*m) for m in [0.5+1.25+0.75]] #midway through stimMove 

    nrows,ncols=5,2 #plot some trials as examples
    fig,axs=plt.subplots(nrows=nrows,ncols=ncols)
    fig.set_size_inches(18,8)

    results=np.zeros((n_trialsperemotion,4),dtype=float)
    results[:]=np.nan
    for i in range(n_trialsperemotion): 
        #ii = i+n_trialsperemotion-nrows*ncols #i for first few trials, or i+n_trialsperemotion-10 for last few trials
        ts = values[i,:] #time series for one trial. array of size n_frames (80)

        ts_5 = acface_utils.moving_average(ts,window_length=5) #smoothed time series (5 data points = 250ms)
        ts_9 = acface_utils.moving_average(ts,window_length=9) #(9 data points = 450ms)

        grad = np.gradient(ts) #gradient of time series
        grad_5 = np.gradient(ts_5) #gradient of smoothed time series
        grad_9=np.gradient(ts_9)
        #index_gradmax = np.argmax(grad) #index of maximum gradient
        index_9_gradmax = np.argmax(grad_9) #index of maximum gradient of smoothed time series
        #grad_smoo5 = acface_utils.moving_average(grad,window_length=5) #smoothed gradient
        #index_gradsmoomax = np.argmax(grad_smoo5) #index of maximum of smoothed gradient

        #dx2=np.gradient(grad) #second derivative
        ts_5_dx2 = np.gradient(grad_5)
        #ts_9_dx2 = np.gradient(grad_9)
        #dx2_smoo5 = acface_utils.moving_average(dx2,window_length=5)
        #dx2_smoo9 = acface_utils.moving_average(dx2,window_length=9)

        #AU intensity should be higher during stimMov and 0.25s after the estimated 'action time', compared to at trigger point or 0.25s before the estimated 'action time'. If these conditions are met, then this counts as a valid facial response. Indicate valid trials with thick grey line        
        index_est = index_9_gradmax #estimated 'action time' is when gradient of smoothed time series is highest
        index_est_minus = index_est - int(0.3 * target_fps) #0.25s before estimated 'action time'
        index_est_plus = index_est + int(0.3 * target_fps) #0.25s after

        if i < nrows*ncols:
            jitter=0.007
            ax = axs[np.unravel_index(i,(nrows,ncols))]
            ax.plot(times_trial_regular,ts,color='black')
            ax.set_title(f'Trial {i}')  
            if np.min(values) >= 0:
                ax.set_ylim(0,3)
            #for j in relevant_timestamps: ax.axvline(x=j,color='black')         
            for k,annotation in enumerate(relevant_labels):
                ax.text(midpoint_timestamps[k],np.min(ts),annotation,ha='center')
            ax.axvline(x=times_trial_regular[index_est]+0*jitter,color='black',linestyle='dashed')

        if (index_est_plus <= len(ts)-1 and index_est_minus >= 0): #check that the estimated peak is not too close to start or end
            indices_lower = default_indices_lower+[index_est_minus]
            indices_upper = default_indices_upper+[index_est_plus]

            if i < nrows*ncols:
                for j in indices_lower: ax.axvline(x=times_trial_regular[j],color='orange')
                for j in indices_upper: ax.axvline(x=times_trial_regular[j],color='blue')
            vals_indices_lower = [ts[m] for m in indices_lower] 
            vals_indices_upper = [ts[m] for m in indices_upper]

            #assert(default_indices_upper[0] > index_est_plus) #check that index_est_plus is before middle of stimMove
            #assert(max(indices_lower) < min(indices_upper)) #that max_group's values are all larger than all of min_group's values

            lower_max = max(acface_utils.get_range(ts,indices_lower)) #max value between trigger and index_est_minus
            upper_min = min(acface_utils.get_range(ts,indices_upper)) #min value between index_est_plus and middle of stimMov
            if lower_max < upper_min: #check that max_group's values are all larger than all of min_group's values
                if i < nrows*ncols: 
                    ax.axvline(0.1,color='grey',linewidth=30,alpha=0.5) #grey bar indicates valid facial response in the trial
                
                #from estimated point of maximum gradient (index_est), move leftward along smoothed time series (ts_9) until values stop decreasing. This is a first estimate of facial response initiation (index_left). Then, move right from this first estimate until you find a local maximum in the 2nd derivative of the smoothed time series (ts_5_dx2). This index_left2 is the final estimate. Do the same 2 steps rightward to find response termination point. Also, make sure that final estimates don't cross index_est again.
                index_left = acface_utils.find_closest_peakdip(ts_9,index_est,'dip','left',hillclimb=True)
                if index_left is None: index_left = int(target_fps*.25) #assuming 250ms response time minimum
                index_right = acface_utils.find_closest_peakdip(ts_9,index_est,'peak','right',hillclimb=True)
                index_left2 = acface_utils.find_closest_peakdip(ts_5_dx2,index_left,'peak','right',hillclimb=False)
                index_right2 = acface_utils.find_closest_peakdip(ts_5_dx2,index_right,'dip','left',hillclimb=False)
                if index_left2 > index_est: index_left2 = index_left
                if index_right2 < index_est: index_right2 = index_right
                """
                index_left = acface_utils.find_closest_peakdip(dx2_smoo5,index_est,'peak','left')
                if index_left is None: index_left = int(target_fps*.25) #assuming 250ms response time minimum
                index_right = acface_utils.find_closest_peakdip(dx2_smoo5,index_est,'dip','right')
                index_left2 = index_left + np.argmax(ts_5_dx2[index_left:index_est])
                index_right2 = index_est + np.argmin(ts_5_dx2[index_est:index_right])
                """

                response_start = times_trial_regular[index_left2]
                response_midtime = times_trial_regular[index_est]
                response_end = times_trial_regular[index_right2]
                maximum_gradient = np.max(grad_5[index_left2:index_right2])
                results[i,:] = [response_start,response_midtime,response_end,maximum_gradient]

                if i < nrows*ncols:
                    ax.axvline(x=times_trial_regular[index_left]+0*jitter,color='green') #initial estimate of response initiation
                    ax.axvline(x=times_trial_regular[index_right]+0*jitter,color='red') #initial estimate of response termination
                    ax.axvline(x=times_trial_regular[index_left2]+1*jitter,color='green',linestyle='dashed') #final estimate of response initiation
                    ax.axvline(x=times_trial_regular[index_right2]+1*jitter,color='red',linestyle='dashed') #final estimate of response termination

    if i < nrows*ncols: fig.tight_layout()

    plt.show()

    assert(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot]))

    """Plotting"""

    '''
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
    '''
    

    


plt.show()
