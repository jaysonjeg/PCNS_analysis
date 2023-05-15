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
from sklearn.decomposition import PCA
import warnings

target_fps=20
ntrials=80
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

    #Get the OpenFace intermediates .csv for this subject
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

    #Get behavioural data for cface non-MRI task 
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\cface1*Ta_H*\\") #'cface' task non-MRI folder for this particular subject
    assert(len(contents)==1) #make sure exactly one matching file
    resultsFolder=contents[0]
    df=pd.read_csv(glob(f"{resultsFolder}*out.csv")[0]) # make log csv into dataframe
    face_summary=pd.read_csv(glob(f"{resultsFolder}*face.csv")[0]) # make face summary csv into dataframe 
    face_summary = {i[0]:i[1] for i in face_summary.values} #convert face summary array into dictionary
    camtstart = face_summary['camtstart'] #time when webcam started recording
    camactualfps = face_summary['camactualfps'] #actual fps of webcam recording

    """
    From out.csv log file, get all pairs of (time,framenumber). Use this to calculate a time since onset for each frame number. Then use this to resample the action unit time series, so that it is indexed by time instead of frame number. 
    """
    event_names = ['instruct','postinstruct','trigger','posttrigger','stimMove','fixation'] #names of events in log file
    times = np.zeros(1+ntrials*len(event_names),dtype='float')
    frames = np.zeros(1+ntrials*len(event_names),dtype='int')
    times[0]=0 #more accurately should be camtstart but negligible difference
    frames[0]=1
    for i in range(len(event_names)):
        event_name = event_names[i]
        event_frames = df[f'{event_name}_camNframe'].values #frame numbers when instruction appeared
        if event_name=='posttrigger': event_name = 'postTrigger' # to account for slight mis-spelling in the column names
        event_onsets = df[f'{event_name}_onset'].values #time when instruction appeared
        start_index = i*ntrials+1
        end_index = start_index + ntrials
        times[start_index:end_index] = event_onsets
        frames[start_index:end_index] = event_frames
    f = interp1d(frames, times,kind='linear',fill_value = 'extrapolate')
    frametimes = f(all_frames) #get time since onset for each frame number, corresponding to rows of aus

    
    def plot_this_au(ax,aus,this_AU='AU12'):
        #Plot AU12 time series with annotations. Blue lines are happy trigger. Red lines are angry trigger
        ax.set_title(this_AU)
        #ax.plot(frametimes,aus_c[this_AU],color='black')
        ax.plot(frametimes,aus[this_AU],color='green')
        #ax.set_ylim(bottom=0)
        ax.set_xlim(left=0,right=530) #max 530     
        for i in df['trigger_onset'][df.ptemot=='HA'].values:
            ax.axvline(x=i,color='mediumpurple') 
        for i in df['trigger_onset'][df.ptemot=='AN'].values:
            ax.axvline(x=i,color='red') 
        """
        for i in df['instruct_onset'][df.ptemot=='HA'].values:
            ax.axvline(x=i,color='darkviolet')
        for i in df['instruct_onset'][df.ptemot=='AN'].values:
            ax.axvline(x=i,color='darkred')
        for i in df['stimMove_onset'][df.ptemot=='HA'].values:
            ax.axvline(x=i,color='blue')
        for i in df['stimMove_onset'][df.ptemot=='AN'].values:
            ax.axvline(x=i,color='darkorange')
        for i in df['fixation_onset'][df.ptemot=='HA'].values:
            ax.axvline(x=i,color='lightblue')
        for i in df['fixation_onset'][df.ptemot=='AN'].values:
            ax.axvline(x=i,color='orange')
        """
    def plot_all_aus(aus):
        fig,axs=plt.subplots(nrows=4,ncols=4)
        fig.set_size_inches(18,8)
        for i in range(n_aus):
            ax = axs[np.unravel_index(i,(4,4))]
            plot_this_au(ax,aus,aus.columns[i])
        fig.tight_layout()

    aus_trigger2stimStop_HA=acface_utils.find_vals_between(df['trigger_camNframe'][df.ptemot=='HA'].values, df['fixation_camNframe'][df.ptemot=='HA'].values, aus) #all action unit values, between trigger and stimStop, when participants were asked to smile (HA)
    aus_trigger2stimStop_HA_max = np.array([np.max(i) for i in aus_trigger2stimStop_HA])
    aus_trigger2stimStop_HA_min = np.array([np.min(i) for i in aus_trigger2stimStop_HA])
    aus_trigger2stimStop_HA_range = aus_trigger2stimStop_HA_max - aus_trigger2stimStop_HA_min
    this_au_trigger2stimStop_HA_max = aus_trigger2stimStop_HA_max[:,this_AU_index] #RETURN
    this_au_trigger2stimStop_HA_range = aus_trigger2stimStop_HA_range[:,this_AU_index] #RETURN

    """
    Get mean time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit, and for first principal component. Could use these as fMRI regressor
    In detail: Get timestamps and AU values for each trial, from trigger to next Instruct. Set timestamps to be relative to the trigger as 0. Concatenate all trials together. Find a linear interpolation of values, and resample at 20 fps to find mean time series.
    """
    start_times = df['trigger_onset'][df.ptemot=='HA'].values
    relevant_timestamps=[0,0.5,0.75,1.5,0.75] #(0, trigger, post_trigger, stimMove, fixation)
    trigger2trialend_secs = sum(relevant_timestamps)
    end_times = [i+trigger2trialend_secs for i in start_times]

    times_pertrial = np.empty(len(start_times),dtype='object') #holds timestamps for each trial
    values_pertrial = np.empty(len(start_times),dtype='object') #array (ntrials) of arrays (ntimepoints*nAUs) containing AU time series for that trial
    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]
        included = (frametimes>=start) & (frametimes<=end)
        value = aus[included] #AU values for all times between 'start' and 'end'
        frametimes_subset = frametimes[included] - start #timestamps corresponding to these AUs, where 'start' is set to 0
        times_pertrial[i] = frametimes_subset
        values_pertrial[i] = value  
    times_alltrials = np.concatenate(times_pertrial)
    values_alltrials = np.vstack(values_pertrial) 
    f = interp1d(times_alltrials, values_alltrials, axis=0, kind='linear',fill_value = 'extrapolate')
    sample_times = np.arange(0,trigger2trialend_secs,1/target_fps)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        values_resampled = f(sample_times) #array (nframes*nAUs) containing mean AU time series, based on all trials concatenated, then resampled
    zerotime_indices = np.where(times_alltrials==0)[0] #manually replace nans at time zero with the mean of values at that time
    zerotime_values = values_alltrials[zerotime_indices,:].mean(axis=0)
    values_resampled[0,:]=zerotime_values #RETURN (but looks weird)

    fig,axs=plt.subplots(nrows=4,ncols=4)
    fig.set_size_inches(18,8)
    for i in range(15): #plot examples of time series from a few trials
        ax = axs[np.unravel_index(i,(4,4))]
        ax.plot(times_pertrial[-i],values_pertrial[-i][this_AU].values,color='blue')
        ax.set_ylim(0,3)
        for j in np.cumsum(relevant_timestamps):
            ax.axvline(x=j) 
    ax=axs[-1,-1]
    fig.tight_layout()
    ax.scatter(times_alltrials,values_alltrials[:,this_AU_index],color='b',marker='.') #plot all trials' time series
    ax.plot(sample_times,values_resampled[:,this_AU_index],color='r') #plot mean time series
    ax.set_ylim(0,3)
    for j in np.cumsum(relevant_timestamps):
        ax.axvline(x=j) 


    #PCA of action unit time series between trigger and stimStop when participants were asked to smile (HA)
    array = np.vstack(aus_trigger2stimStop_HA)
    pca = PCA()
    pca.fit(array)
    comp0 = pca.components_[0]
    aus_pca = pca.transform(aus)
    aus_pca = pd.DataFrame(aus_pca)
    aus_pca.columns = [f'comp{i}' for i in range(aus_pca.shape[1])]   
    assert(acface_utils.pca_comp0_direction_correct(target_fps,values_resampled,pca))


    #Plot to see whether response amplitudes are decreasing with time
    """
    plt.scatter(range(len(this_au_trigger2stimStop_HA_max)) , this_au_trigger2stimStop_HA_max) 
    plt.xlabel('Trial')
    plt.ylabel('Max AU12 value from trigger to stimStop')
    """

    
    fig,ax=plt.subplots()
    fig.set_size_inches(18,3) #40,5
    #plot_this_au(ax,aus,this_AU='AU12')
    #plot_all_aus(aus)
    plot_this_au(ax,aus_pca,this_AU='comp0')
    #plot_all_aus(aus_pca)
    #print([[aus_names[i],comp0[i]] for i in range(len(comp0))] )

    # Plot how PCA components map onto action units
    """
    fig, ax = plt.subplots()
    im = ax.imshow(pca.components_.T)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('AU')
    ax.set_yticks(np.arange(len(aus_names)))
    ax.set_yticklabels(aus_names)
    ax.set_xticks(np.arange(len(aus_pca.columns)))
    ax.set_xlabel('Component')
    ax.set_xticklabels([i[-1] for i in aus_pca.columns])
    """



plt.show()
