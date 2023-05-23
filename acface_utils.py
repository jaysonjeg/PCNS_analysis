import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import warnings
import acommon

def plot_this_au(df,ax,times,aust,this_AU='AU12',color='green',label=None):
    #Plot AU12 time series with annotations. Blue lines are happy trigger. Red lines are angry trigger
    ax.set_title(this_AU)
    ax.plot(times,aust[this_AU],color=color,label=label)
    ax.set_xlim(left=0,right=530) #max 530  
    ax.legend()   
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

def plot_this_au_trial(values,title,times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps):
    #plot single-AU time series for a few trials, each in a separate panel
    #values should be array size ntrials (40) * nframes (80)
    fig,axs=plt.subplots(nrows=4,ncols=4)
    fig.set_size_inches(18,8)
    for i in range(16): 
        ax = axs[np.unravel_index(i,(4,4))]
        ax.plot(times_trial_regular,values[i,:],color='blue')
        ax.set_title(f'Trial {i}')
        if np.min(values) >= 0:
            ax.set_ylim(0,3)
        for j in relevant_timestamps:
            ax.axvline(x=j) 
        for i,annotation in enumerate(relevant_labels):
            ax.text(midpoint_timestamps[i],+0.5,annotation,ha='center')
    fig.suptitle(title)
    fig.tight_layout()

def plot_this_au_trial_superimposed(emot,index,title,aus_trial,aus_trial_mean,aus_trial_mean2,relevant_timestamps,relevant_labels,midpoint_timestamps,times_trial_regular):
    #On a single plot, plot each trial's single-AU time series in blue, then mean time course in black
    fig,ax=plt.subplots()
    for i in range(aus_trial[emot].shape[0]):
        ax.plot(times_trial_regular,aus_trial[emot][i,:,index],color='blue',linewidth=0.2)
    ax.plot(times_trial_regular,aus_trial_mean[emot][:,index],color='k',linewidth=1,label='mean') #plot mean time series
    ax.plot(times_trial_regular,aus_trial_mean2[emot][:,index],color='r',linewidth=1,label='mean2') #plot mean time series, alternate method
    ax.legend()
    ax.set_title(title)
    #ax.set_ylim(0,3)
    for j in relevant_timestamps:
        ax.axvline(x=j) 
    for i,annotation in enumerate(relevant_labels):
        ax.text(midpoint_timestamps[i],-0.5,annotation,ha='center')
    fig.tight_layout()


def find_duplicate_indices(lst):
    #In list lst, for each item that appears more than once, return the indices of all its appearances after the first appearance
    seen = set()
    duplicate_indices = []
    for index, item in enumerate(lst):
        if item in seen:
            duplicate_indices.append(index)
        else:
            seen.add(item)
    return duplicate_indices

def moving_average(x,window_length):
    #window_length should be odd number. Pads left and right with values on the end
    assert(window_length%2==1)
    kernel = np.ones(window_length) / window_length
    temp = np.convolve(x,kernel,'valid')
    n = int((window_length-1)/2)
    result = np.zeros(len(x),dtype=float)
    result[0:n] = [temp[0]]*n
    result[-n:] = [temp[-1]]*n
    result[n:-n] = temp
    return result

def find_closest_peakdip(x, t,peak_or_dip,left_or_right,hillclimb=True):
    #Given time series x and index t, find index of closest peak to the right of t
    if left_or_right=='right':
        iterator = range(t,len(x))
    elif left_or_right=='left':
        iterator = range(t,0,-1)
    for n in iterator:
        if hillclimb:
            current=x[n]
            if left_or_right=='right':
                comparator = x[n+1]
            elif left_or_right=='left':
                comparator = x[n-1]
            if peak_or_dip=='peak' and comparator < current:
                return n
            elif peak_or_dip=='dip' and comparator > current:
                return n
        else:
            if peak_or_dip=='peak':
                if x[n] > x[n-1] and x[n] > x[n+1]:
                    return n
            if peak_or_dip=='dip':
                if x[n] < x[n-1] and x[n] < x[n+1]:
                    return n

    return None #no dip found

def rescale(x):
    min_val = np.min(x)
    max_val = np.max(x)
    rescaled_x = (x - min_val) / (max_val - min_val)
    return rescaled_x 

def get_range(vals,indices):
    p,q=min(indices),max(indices)
    if p==q: return vals[p]
    else: return vals[p:q]
    
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

def get_pca(list_of_arrays):
    array = np.vstack(list_of_arrays)
    pca=PCA(n_components = acommon.n_aus)
    pca.fit(array)
    return pca


def pca_transform(pca,values):
    result=pca.transform(values)
    result=pd.DataFrame(result)
    result.columns = [f'comp{i}' for i in range(result.shape[1])]
    return result

def pca_comp0_direction_correct(target_fps,values_resampled,pca):
    #Check the direction of the first principal component. Return whether it increases from trigger onset to middle of stimMove
    mid_stimMove_frame = target_fps * 2 #approximate no of frames from trigger to middle of stimMove
    values_resampled_0_to_mid_stimMove = values_resampled[mid_stimMove_frame,:] - values_resampled[0,:] 
    comp0_0_to_mid_stimMove = values_resampled_0_to_mid_stimMove @ pca.components_[0]
    return comp0_0_to_mid_stimMove > 0 

def calculate_averages(lst):
    #Given a list of numbers, return a list of the averages of consecutive pair of numbers
    return [np.mean([lst[i],lst[i+1]]) for i in range(len(lst)-1)]

def get_all_post_trigger_time_series(df,interp_aus,times_trial_regular,emotion='ha'):
    start_times = df['trigger_onset'][df.ptemot==emotion.upper()].values
    times_pertrial = np.array([times_trial_regular for i in range(len(start_times))]) #with 0 at each trigger
    times_pertrial_true = np.array([start_time + times_trial_regular for start_time in start_times]) #holds resampled timestamps for each trial
    aus_trial = np.array([interp_aus(times) for times in times_pertrial_true]) #AU values for all times between 'start' and 'end'  
    times_alltrials = np.concatenate(times_pertrial)
    #times_alltrials = np.tile(times_trial_regular,len(start_times))
    return aus_trial

def get_mean_post_trigger_time_series(times_trial_regular,aus_trial):
    """
    Another way of finding the 'mean time series for each trial'. Concatenate data from all trials, then interpolate from this combined data
    """
    values_alltrials = np.vstack(aus_trial) 
    times_alltrials = np.tile(times_trial_regular,len(aus_trial))
    f = interp1d(times_alltrials, values_alltrials, axis=0, kind='linear',fill_value = 'extrapolate')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        values_resampled = f(times_trial_regular) #array (nframes*nAUs) containing mean AU time series, based on all trials concatenated, then resampled
    zerotime_indices = np.where(times_alltrials==0)[0] #manually replace nans at time zero with the mean of values at that time
    zerotime_values = values_alltrials[zerotime_indices,:].mean(axis=0)
    values_resampled[0,:]=zerotime_values #RETURN (but looks weird)
    return values_resampled