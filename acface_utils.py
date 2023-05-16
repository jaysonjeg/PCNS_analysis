import numpy as np
from matplotlib import pyplot as plt
import warnings
from scipy.interpolate import interp1d

def plot_this_au(df,ax,times,aust,this_AU='AU12'):
    #Plot AU12 time series with annotations. Blue lines are happy trigger. Red lines are angry trigger
    ax.set_title(this_AU)
    ax.plot(times,aust[this_AU],color='green')
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
def plot_all_aus(df,times,aust):
    fig,axs=plt.subplots(nrows=4,ncols=4)
    fig.set_size_inches(18,8)
    for i in range(aust.shape[1]):
        ax = axs[np.unravel_index(i,(4,4))]
        plot_this_au(df,ax,times,aust,aust.columns[i])
    fig.tight_layout()
    
def get_all_timestamps_and_framestamps(df,ntrials):
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
    return times,frames

def find_vals_between(start_frames,end_frames,aus):
#For each pairs of values in start_frames and end_frames. Return all rows in aus whose indices are between the lower and upper of these values.
    values=np.empty(len(start_frames),dtype='object')
    for i in range(len(start_frames)):
        start=start_frames[i]
        end=end_frames[i]
        values[i] = aus.iloc[start:end,:]
    return values
def pca_comp0_direction_correct(target_fps,values_resampled,pca):
    #Check the direction of the first principal component. Return whether it increases from trigger onset to middle of stimMove
    mid_stimMove_frame = target_fps * 2 #approximate no of frames from trigger to middle of stimMove
    values_resampled_0_to_mid_stimMove = values_resampled[mid_stimMove_frame,:] - values_resampled[0,:] 
    comp0_0_to_mid_stimMove = values_resampled_0_to_mid_stimMove @ pca.components_[0]
    return comp0_0_to_mid_stimMove > 0 

def calculate_averages(lst):
    #Given a list of numbers, return a list of the averages of consecutive pair of numbers
    return [np.mean([lst[i],lst[i+1]]) for i in range(len(lst)-1)]

def get_mean_post_trigger_time_series(df,interp_aus,target_fps,relevant_timegaps,emotion='HA'):
    trigger2trialend_secs = sum(relevant_timegaps)
    times_trial_regular = np.arange(0,trigger2trialend_secs,1/target_fps) #new timestamps for resampling
    start_times = df['trigger_onset'][df.ptemot==emotion].values
    times_pertrial = np.array([times_trial_regular for i in range(len(start_times))]) #with 0 at each trigger
    times_pertrial_true = np.array([start_time + times_trial_regular for start_time in start_times]) #holds resampled timestamps for each trial
    values_pertrial = np.array([interp_aus(times) for times in times_pertrial_true]) #AU values for all times between 'start' and 'end'  
    times_alltrials = np.concatenate(times_pertrial)
    #times_alltrials = np.tile(times_trial_regular,len(start_times))
    values_alltrials = np.vstack(values_pertrial) 
    values_pertrial_mean = np.mean(values_pertrial,axis=0) #mean AU time series for each trial
    f = interp1d(times_alltrials, values_alltrials, axis=0, kind='linear',fill_value = 'extrapolate')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        values_resampled = f(times_trial_regular) #array (nframes*nAUs) containing mean AU time series, based on all trials concatenated, then resampled
    zerotime_indices = np.where(times_alltrials==0)[0] #manually replace nans at time zero with the mean of values at that time
    zerotime_values = values_alltrials[zerotime_indices,:].mean(axis=0)
    values_resampled[0,:]=zerotime_values #RETURN (but looks weird)
    return times_pertrial,values_pertrial,times_trial_regular,values_resampled,values_pertrial_mean

def get_mean_post_trigger_time_series2(df,aus,target_fps,relevant_timegaps,emotion='HA'):
    """
    As above, but without using pre-interpolated data
    """
    start_times = df['trigger_onset'][df.ptemot==emotion].values
    trigger2trialend_secs = sum(relevant_timegaps)
    end_times = [i+trigger2trialend_secs for i in start_times]
    times_pertrial = np.empty(len(start_times),dtype='object') #holds timestamps for each trial
    values_pertrial = np.empty(len(start_times),dtype='object') #array (ntrials) of arrays (ntimepoints*nAUs) containing AU time series for that trial
    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]
        included = (times_eachframe>=start) & (times_eachframe<=end)
        times_pertrial[i] = times_eachframe[included] - start #timestamps corresponding to these AUs, where 'start' is set to 0
        values_pertrial[i] = aus[included] #AU values for all times between 'start' and 'end'  
    times_alltrials = np.concatenate(times_pertrial)
    values_alltrials = np.vstack(values_pertrial) 
    f = interp1d(times_alltrials, values_alltrials, axis=0, kind='linear',fill_value = 'extrapolate')
    times_trial_regular = np.arange(0,trigger2trialend_secs,1/target_fps)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        values_resampled = f(times_trial_regular) #array (nframes*nAUs) containing mean AU time series, based on all trials concatenated, then resampled
    zerotime_indices = np.where(times_alltrials==0)[0] #manually replace nans at time zero with the mean of values at that time
    zerotime_values = values_alltrials[zerotime_indices,:].mean(axis=0)
    values_resampled[0,:]=zerotime_values #RETURN (but looks weird)
    return times_pertrial,values_pertrial,times_trial_regular,values_resampled