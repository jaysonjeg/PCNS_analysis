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

def plot_this_au_trial(values,title,times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps,plot_relevant_timestamps=True,results=None):
    #plot single-AU time series for a few trials, each in a separate panel
    #values should be array size ntrials (40) * nframes (80)
    nrows,ncols=5,2
    fig,axs=plt.subplots(nrows=nrows,ncols=ncols)
    fig.set_size_inches(18,8)
    for i in range(nrows*ncols): 
        ii = i #i+n_trialsperemotion-nrows*ncols #i for first few trials, or i+n_trialsperemotion-10 for last few trials
        ts = values[ii,:] #time series for one trial. array of size n_frames (80)
        ax = axs[np.unravel_index(i,(nrows,ncols))]
        ax.plot(times_trial_regular,ts,color='black')
        ax.set_title(f'Trial {ii}')
        if np.min(values) >= 0:
            ax.set_ylim(0,3)
        if plot_relevant_timestamps:
            for j in relevant_timestamps:
                ax.axvline(x=j) 
        for i,annotation in enumerate(relevant_labels):
            ax.text(midpoint_timestamps[i],np.min(ts),annotation,ha='center')

        if results is not None:
            jitter=0.007            
            ax.axvline(x=results.mid[ii]+0*jitter,color='black',linestyle='dashed')
            if not(np.isnan(results.times_lower[ii]).all()):
                for j in results.times_lower[ii]: ax.axvline(x=j,color='orange')
                for j in results.times_upper[ii]: ax.axvline(x=j,color='blue')
            if not(np.isnan(results.indices_leftright[ii]).all()):
                ax.axvline(0.1,color='grey',linewidth=30,alpha=0.5) #grey bar indicates valid facial response in the trial
                index_left,index_right = results.indices_leftright[ii]
                index_left2 = results.start[ii]
                index_right2 = results.end[ii]
                ax.axvline(x=index_left+0*jitter,color='green') #initial estimate of response initiation
                ax.axvline(x=index_right+0*jitter,color='red') #initial estimate of response termination
                ax.axvline(x=index_left2+1*jitter,color='green',linestyle='dashed') #final estimate of response initiation
                ax.axvline(x=index_right2+1*jitter,color='red',linestyle='dashed') #final estimate of response termination


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


def get_latency(values,target_fps,n_trialsperemotion,times_trial_regular,plot=False):
    """
    Function to find facial response's initiation and termination points, to calculate latency and duration
    Also outputs maximum slope of the facial response
    values is a resampled time series for each trial, for a single AU: n_trialsperemotion (40) * n_frames (80) 
    """
    def get_range(vals,indices):
        p,q=min(indices),max(indices)
        if p==q: return [vals[p]]
        else: return vals[p:q]

    #Timepoints to compare to make sure valid facial response was detected
    default_indices_lower = [5] #250ms after trigger (optionally add at start of nextInstruct to make sure they look normal again)
    default_indices_upper = [int(target_fps*m) for m in [0.5+1.25+0.75]] #midway through stimMove 

    if plot:

        relevant_timejitters=[0,0.5,0.75,1.5,0.75,0.5] #(0, trigger, post_trigger, stimMove, fixation, next instruction)
        relevant_labels=['trigger','post_trigger','stimMove','fixation','next_instruct']
        relevant_timestamps = np.cumsum(relevant_timejitters)
        midpoint_timestamps = calculate_averages(relevant_timestamps)   

        nrows,ncols=5,2 #plot some trials as examples
        fig,axs=plt.subplots(nrows=nrows,ncols=ncols)
        fig.set_size_inches(18,8)
        

    results = pd.DataFrame(data=None,index=range(n_trialsperemotion), columns=['start','mid','end','max_grad','times_lower','times_upper','indices_leftright'],dtype=object)

    for i in range(n_trialsperemotion): 
        #ii = i+n_trialsperemotion-nrows*ncols #i for first few trials, or i+n_trialsperemotion-10 for last few trials
        ts = values[i,:] #time series for one trial. array of size n_frames (80)

        ts_5 = moving_average(ts,window_length=5) #smoothed time series (5 data points = 250ms)
        ts_9 = moving_average(ts,window_length=9) #(9 data points = 450ms)

        grad = np.gradient(ts) #gradient of time series
        grad_5 = np.gradient(ts_5) #gradient of smoothed time series
        grad_9=np.gradient(ts_9)
        #index_gradmax = np.argmax(grad) #index of maximum gradient
        index_9_gradmax = np.argmax(grad_9) #index of maximum gradient of smoothed time series
        #grad_smoo5 = moving_average(grad,window_length=5) #smoothed gradient
        #index_gradsmoomax = np.argmax(grad_smoo5) #index of maximum of smoothed gradient

        #dx2=np.gradient(grad) #second derivative
        ts_5_dx2 = np.gradient(grad_5)
        #ts_9_dx2 = np.gradient(grad_9)
        #dx2_smoo5 = moving_average(dx2,window_length=5)
        #dx2_smoo9 = moving_average(dx2,window_length=9)

        #AU intensity should be higher during stimMov and 0.25s after the estimated 'action time', compared to at trigger point or 0.25s before the estimated 'action time'. If these conditions are met, then this counts as a valid facial response. Indicate valid trials with thick grey line        
        index_est = index_9_gradmax #estimated 'action time' is when gradient of smoothed time series is highest
        index_est_minus = index_est - int(0.3 * target_fps) #0.25s before estimated 'action time'
        index_est_plus = index_est + int(0.3 * target_fps) #0.25s after
        
        results.mid[i] = times_trial_regular[index_est]
        
        if plot and i < nrows*ncols:
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
            times_lower = [times_trial_regular[m] for m in indices_lower]
            times_upper = [times_trial_regular[m] for m in indices_upper]

            results.times_lower[i] = times_lower
            results.times_upper[i] = times_upper

            if plot and  i < nrows*ncols:
                for j in times_lower: ax.axvline(x=j,color='orange')
                for j in times_upper: ax.axvline(x=j,color='blue')

            #assert(default_indices_upper[0] > index_est_plus) #check that index_est_plus is before middle of stimMove
            #assert(max(indices_lower) < min(indices_upper)) #that max_group's values are all larger than all of min_group's values

            lower_max = max(get_range(ts,indices_lower)) #max value between trigger and index_est_minus
            upper_min = min(get_range(ts,indices_upper)) #min value between index_est_plus and middle of stimMov
            if lower_max < upper_min: #check that max_group's values are all larger than all of min_group's values
                
                if plot and i < nrows*ncols: 
                    ax.axvline(0.1,color='grey',linewidth=30,alpha=0.5) #grey bar indicates valid facial response in the trial
                
                #from estimated point of maximum gradient (index_est), move leftward along smoothed time series (ts_9) until values stop decreasing. This is a first estimate of facial response initiation (index_left). Then, move right from this first estimate until you find a local maximum in the 2nd derivative of the smoothed time series (ts_5_dx2). This index_left2 is the final estimate. Do the same 2 steps rightward to find response termination point. Also, make sure that final estimates don't cross index_est again.
                index_left = find_closest_peakdip(ts_9,index_est,'dip','left',hillclimb=True)
                if index_left is None: index_left = int(target_fps*.25) #assuming 250ms response time minimum
                index_right = find_closest_peakdip(ts_9,index_est,'peak','right',hillclimb=True)
                index_left2 = find_closest_peakdip(ts_5_dx2,index_left,'peak','right',hillclimb=False)
                index_right2 = find_closest_peakdip(ts_5_dx2,index_right,'dip','left',hillclimb=False)
                if index_left2 > index_est: index_left2 = index_left
                if index_right2 < index_est: index_right2 = index_right
                """
                index_left = acface_utils.find_closest_peakdip(dx2_smoo5,index_est,'peak','left')
                if index_left is None: index_left = int(target_fps*.25) #assuming 250ms response time minimum
                index_right = acface_utils.find_closest_peakdip(dx2_smoo5,index_est,'dip','right')
                index_left2 = index_left + np.argmax(ts_5_dx2[index_left:index_est])
                index_right2 = index_est + np.argmin(ts_5_dx2[index_est:index_right])
                """

                results.indices_leftright[i] = [times_trial_regular[index_left],times_trial_regular[index_right]]
                results.max_grad[i] = np.max(grad_5[index_left2:index_right2])
                results.start[i] = times_trial_regular[index_left2]
                results.end[i] = times_trial_regular[index_right2]

                
                if plot and i < nrows*ncols:
                    ax.axvline(x=times_trial_regular[index_left]+0*jitter,color='green') #initial estimate of response initiation
                    ax.axvline(x=times_trial_regular[index_right]+0*jitter,color='red') #initial estimate of response termination
                    ax.axvline(x=times_trial_regular[index_left2]+1*jitter,color='green',linestyle='dashed') #final estimate of response initiation
                    ax.axvline(x=times_trial_regular[index_right2]+1*jitter,color='red',linestyle='dashed') #final estimate of response termination
                
    if plot: 
        fig.tight_layout()
        plt.show()

    return results

def extract_subject_result(r,n_trialsperemotion):
    #Given results dataframe from function acface_utils.get_latency, extract the following:
    r_valid = [not(np.isnan(j).all()) for j in r.indices_leftright]
    r_validperc = 100 * np.sum(r_valid) / n_trialsperemotion #valid percentage
    r_latencies = [r.start[i] for i in range(n_trialsperemotion) if r_valid[i]] #OUTCOME 2
    r_durations = [r.end[i] - r.start[i] for i in range(n_trialsperemotion) if r_valid[i]]
    r_maxgrads = [r.max_grad[i] for i in range(n_trialsperemotion) if r_valid[i]] #OUTCOME 5
    return r_validperc,r_latencies,r_durations,r_maxgrads