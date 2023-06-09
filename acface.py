"""
Analyse cface1 face data
Mainly using the out.csv log file, and the OpenFace-processed .csv file with action unit time series
Within each trial, key events are 'trigger', 'stimMove', 'stimStop', 'fixation'
Resample action unit series from being indexed by frames, to be indexed by time (sec)

Get the following metrics of facial movement for each subject:
1) amplitudes: Amplitude of smile or frown action post-trigger. Just use max(AU12) and (max(A12) - preTrigger(AU12)). Does this decrease with time due to tiredness?
2) mean_ts: Mean time series from trigger to next Instruct, averaged across trials, when they're asked to smile (HA) or frown(AN) separately. Get this for each action unit. Could use these as fMRI regressor
3) mean_ts_pca: Point 3 but for first principal component
4) latencies: Distribution of latency post-trigger for the smile or frown action. Look at distribution and exclude outliers
5) maxgrad: Maximum value of first derivative for AU12 or PCA component

Deal with amplitude as confounder

Issues:
- Many trials have no response. No sudden uptick. So average time series may not be accurate
- Even trials with a response are not well modelled by average time series, which is a smooth curve. This is because the average time series is a smooth curve, whereas the actual time series is more like a step function.

For fMRI, options are:
 - regress out subject-specific mean time series
 - regress out subject-specific 'start' (latency) and 'end' times in a box function (but this doesn't deal with the expression offset)
 - see if fMRI contrasts are associated with amplitude, latency, maxgrad
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
get_mean_ts=True
get_mean_ts_pca=True 
get_latencies=True 
get_maxgrads=True
to_plot=False #plots for each participant

t['use_cface'] = ((include) & (t.valid_cfacei==1) & (t.valid_cfaceo==1)) #those subjects whose cface data we will use

#We will later be getting AU time series for each trial, and resampling timestamps to be relative to trigger as 0, and at 20fps (target_fps). Here we new resampled per-trial timestamps
relevant_timejitters=[0,0.5,0.75,1.5,0.75,0.5] #(0, trigger, post_trigger, stimMove, fixation, next instruction)
relevant_labels=['trigger','post_trigger','stimMove','fixation','next_instruct']
relevant_timestamps = np.cumsum(relevant_timejitters)
midpoint_timestamps = acface_utils.calculate_averages(relevant_timestamps)   
times_trial_regular = np.arange(0,relevant_timestamps[-1],1/target_fps) #new timestamps for resampling

def get_outcomes(subject):
    print(c.time()[1])
    all_frames,aus,success = acommonfuncs.get_openface_table('cface1',subject,static_or_dynamic) #Get the OpenFace intermediates .csv for this subject

    webcam_frames_success_proportion = np.sum(success)/len(success)
    if webcam_frames_success_proportion < min_success:
        print(f"WARNING: {subject} has only {webcam_frames_success_proportion:.3f} proportion of successful frames. Returning nan.")
        keys = ['amp_max','amp_range','mean_ts','mean_ts_pca']
        return {key:np.nan for key in keys}

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
    Get interpolated AU time series for each trial separately. Trials are grouped into 2 groups: when they're asked to smile (HA) or frown (AN). In detail: Get timestamps and AU values for each trial, from trigger to next Instruct. Set timestamps to be relative to the trigger as 0. Find a linear interpolation of values, and resample at 20 fps.
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

    

    if get_mean_ts:
        #Get mean (across trials) of the per-trial AU time series when they're asked to smile (HA) or frown(AN) separately. Could use these as fMRI regressor
        aus_trial_mean2,aus_trial_mean={},{} #aus_trial_mean['ha'] has mean across trials, separately for each timepoint
        for emot in emots:
            aus_trial_mean[emot]=np.mean(aus_trial[emot],axis=0) #OUTCOME mean_ts
            aus_trial_mean2[emot] = acface_utils.get_mean_post_trigger_time_series(times_trial_regular,aus_trial[emot]) #alternative method (not good)

    if get_mean_ts_pca:
        #PCA of action unit time series for each emotion separately
        pca,comp0,aus_pca,aus_trial_pca,aust_pca,aus_trial_pca_mean,aus_trial_pca_mean2={},{},{},{},{},{},{}
        for emot in emots:
            pca[emot] = acface_utils.get_pca(aus_trial[emot])
            if not(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot])):
                pca[emot].components_[0] = -pca[emot].components_[0] #ensure component zero increases from trigger to middle of stimMove
            comp0[emot] = pca[emot].components_[0]
            aus_trial_pca[emot] = np.array([pca[emot].transform(i) for i in aus_trial[emot]]) 
            aus_trial_pca_mean[emot] = pca[emot].transform(aus_trial_mean[emot]) #OUTCOME mean_ts_pca
            aus_trial_pca_mean2[emot] = pca[emot].transform(aus_trial_mean2[emot])
            aus_pca[emot] = acface_utils.pca_transform(pca[emot],aus) 
            aust_pca[emot] = acface_utils.pca_transform(pca[emot],aust) 
        #interp_aus_pca = interp1d(times_eachframe,aus_pca['ha'],axis=0,kind='linear',fill_value = 'extrapolate')  
        assert(acface_utils.pca_comp0_direction_correct(target_fps,aus_trial_mean[emot],pca[emot]))

    if get_latencies and get_maxgrads:
        get_latency = lambda values: acface_utils.get_latency(values,target_fps,n_trialsperemotion,times_trial_regular,plot=False) 
        other_metrics= {key:get_latency(aus_trial_pca[key][:,:,0]) for key in emots} #metrics around latency, using first PCA component of smile and frown trials

        #r_an_pca0 = get_latency(aus_trial_pca['an'][:,:,0])
        #r_ha_pca0 = get_latency(aus_trial_pca['ha'][:,:,0])
        #r_ha_AU12 = get_latency(aus_trial_pca['ha'][:,:,ha_AU_index])
        #acface_utils.plot_this_au_trial(r_an_pca0,'sub sz 0 - an - comp0',times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps,plot_relevant_timestamps=False,results=results)

        #r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(r_ha_pca0,n_trialsperemotion)    
        #r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(r_an_pca0,n_trialsperemotion)
        #r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(r_ha_AU12,n_trialsperemotion) 

    """Plotting for single subject"""
    if to_plot:       
        plot_this_au_trial = lambda values,title,results: acface_utils.plot_this_au_trial(values,title,times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps,results=results)
        emot='ha'
        values = aus_trial[emot][:,:,ha_AU_index]
        title = f'{ha_AU} time series for some {emot} trials'
        plot_this_au_trial(values,title,other_metrics['ha'])
        emot='an'
        values = aus_trial_pca[emot][:,:,0]
        title = f'PCA comp 0 time series for some {emot} trials'
        plot_this_au_trial(values,title,other_metrics['an'])

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
        assert(0)

    return {'amp_max':ha_AU_trial_ha_max, 'amp_range':ha_AU_trial_ha_range,'mean_ts': aus_trial_mean,'mean_ts_pca':aus_trial_pca_mean, 'other_metrics':other_metrics}
    #return ha_AU_trial_ha_max


def add_columns(colnames,dtype=object,value=np.nan):
    for colname in colnames:
        t[colname]=value
        t[colname]=t[colname].astype(dtype)
add_columns(['cface_mean_ts_ha_pca0','cface_mean_ts_an_pca0','cface_mean_ts_ha_au12'])
add_columns(['cface_latencies_ha','cface_durations_ha','cface_latencies_an','cface_durations_an'])

for i in range(t.shape[0]):
    if t.use_cface[i]:
        outcomes = get_outcomes(subs[i])
        #t.at[i,'cface_amp_max_mean'] = np.mean(outcomes['amp_max'])
        #t.at[i,'cface_amp_range_mean'] = npt.co.mean(outcomes['amp_range'])
        #t.at[i,'cface_amp_max_slope'] = acface_utils.get_slope(outcomes['amp_max'])
        if type(outcomes['mean_ts_pca'])==dict: #exclude nans from poor webcam acquisitions
            t.at[i,'cface_goodwebcam']=True #whether webcam acquisition was good enough to use

            t.at[i,'cface_amp_max_mean'] = np.mean(outcomes['amp_max'])
            t.at[i,'cface_amp_range_mean'] = np.mean(outcomes['amp_range'])
            t.at[i,'cface_amp_max_slope'] = acface_utils.get_slope(outcomes['amp_max'])

            t.at[i,'cface_mean_ts_ha_pca0'] = outcomes['mean_ts_pca']['ha'][:,0]
            t.at[i,'cface_mean_ts_an_pca0'] = outcomes['mean_ts_pca']['an'][:,0]
            t.at[i,'cface_mean_ts_ha_au12'] = outcomes['mean_ts']['ha'][:,ha_AU_index]
            
            r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(outcomes['other_metrics']['ha'],n_trialsperemotion)
            t.at[i,'cface_latencies_validperc_ha'] = r_validperc
            t.at[i,'cface_latencies_ha'] = r_latencies
            t.at[i,'cface_latencies_mean_ha'] = np.mean(r_latencies)
            t.at[i,'cface_durations_ha'] = r_durations
            t.at[i,'cface_durations_mean_ha'] = np.mean(r_durations)
            t.at[i,'cface_maxgrads_mean_ha'] = np.mean(r_maxgrads)
            r_validperc,r_latencies,r_durations,r_maxgrads=acface_utils.extract_subject_result(outcomes['other_metrics']['an'],n_trialsperemotion)
            t.at[i,'cface_latencies_validperc_an'] = r_validperc
            t.at[i,'cface_latencies_an'] = r_latencies
            t.at[i,'cface_latencies_mean_an'] = np.mean(r_latencies)
            t.at[i,'cface_durations_an'] = r_durations
            t.at[i,'cface_durations_mean_an'] = np.mean(r_durations)
            t.at[i,'cface_maxgrads_mean_an'] = np.mean(r_maxgrads)
            #acface_utils.plot_this_au_trial(outcomes['other_metrics']['an'],'an - comp0',times_trial_regular,relevant_timestamps,relevant_labels,midpoint_timestamps,plot_relevant_timestamps=False,results=outcomes['other_metrics']['an'])          
        else:
            t.at[i,'cface_goodwebcam']=False


import seaborn as sns
from scipy.stats import ttest_ind
show_subs = (hc|sz) & t.use_cface

def sns_plot(**kwargs):
    sns.swarmplot(**kwargs,y='group01',alpha=0.5,palette=colors)#swarmplot, stripplot
    sns.violinplot(**kwargs,y='group01',inner='box',cut=0,color='yellow')
def pval(column_name):
    tstat,p=ttest_ind(t.loc[hc & t.use_cface,column_name].dropna(),t.loc[sz & t.use_cface,column_name].dropna())
    return p

#Look at amplitudes
fig,axs=plt.subplots(2)
sns_plot(ax=axs[0],data=t.loc[show_subs,:], x='cface_amp_max_mean')
#sns_plot(ax=axs[0],data=t.loc[show_subs,:], x='cface_amp_range_mean')
sns_plot(ax=axs[1],data=t.loc[show_subs,:], x='cface_amp_max_slope')
axs[0].set_title(f"p={pval('cface_amp_max_mean'):.2f}")
axs[1].set_title(f"p={pval('cface_amp_max_slope'):.2f}")
fig.tight_layout()

#Look at latencies
fig,axs=plt.subplots(4)
sns_plot(ax=axs[0],data=t.loc[show_subs,:], x='cface_latencies_validperc_ha')
sns_plot(ax=axs[1],data=t.loc[show_subs,:], x='cface_latencies_mean_ha')
sns_plot(ax=axs[2],data=t.loc[show_subs,:], x='cface_durations_mean_ha')
sns_plot(ax=axs[3],data=t.loc[show_subs,:], x='cface_maxgrads_mean_ha')
axs[0].set_title(f"p={pval('cface_latencies_validperc_ha'):.2f}")
axs[1].set_title(f"p={pval('cface_latencies_mean_ha'):.2f}")
axs[2].set_title(f"p={pval('cface_durations_mean_ha'):.2f}")
axs[3].set_title(f"p={pval('cface_maxgrads_mean_ha'):.2f}")
fig.tight_layout()

fig,axs=plt.subplots(4)
sns_plot(ax=axs[0],data=t.loc[show_subs,:], x='cface_latencies_validperc_an')
sns_plot(ax=axs[1],data=t.loc[show_subs,:], x='cface_latencies_mean_an')
sns_plot(ax=axs[2],data=t.loc[show_subs,:], x='cface_durations_mean_an')
sns_plot(ax=axs[3],data=t.loc[show_subs,:], x='cface_maxgrads_mean_an')
axs[0].set_title(f"p={pval('cface_latencies_validperc_an'):.2f}")
axs[1].set_title(f"p={pval('cface_latencies_mean_an'):.2f}")
axs[2].set_title(f"p={pval('cface_durations_mean_an'):.2f}")
axs[3].set_title(f"p={pval('cface_maxgrads_mean_an'):.2f}")
fig.tight_layout()

#Look at mean time series
def plot_mean_ts(ax,data,title,which_subjects,ylims=[-2.5,2.5]):
    for i in range(len(t)):
        if t.use_cface[i] and (hc|sz|sza)[i] and t.cface_goodwebcam[i]: 
            if which_subjects[i]:
                ax.plot(times_trial_regular,data[i],color='black',linewidth=0.4)
    for j in relevant_timestamps:
        ax.axvline(x=j) 
    for i,annotation in enumerate(relevant_labels):
        ax.text(midpoint_timestamps[i],-0.5,annotation,ha='center')
    ax.set_title(title)
    ax.set_ylim(ylims)
fig,axs=plt.subplots(3,3)
plot_mean_ts(axs[0,0],t.cface_mean_ts_ha_au12,'HC - ha - AU12',hc,ylims=[0,4])
plot_mean_ts(axs[1,0],t.cface_mean_ts_ha_au12,'SZA - ha - AU12',sza,ylims=[0,4])
plot_mean_ts(axs[2,0],t.cface_mean_ts_ha_au12,'SZ - ha - AU12',sz,ylims=[0,4])
plot_mean_ts(axs[0,1],t.cface_mean_ts_ha_pca0,'HC - ha - pca0',hc)
plot_mean_ts(axs[1,1],t.cface_mean_ts_ha_pca0,'SZA - ha - pca0',sza)
plot_mean_ts(axs[2,1],t.cface_mean_ts_ha_pca0,'SZ - ha - pca0',sz)
plot_mean_ts(axs[0,2],t.cface_mean_ts_an_pca0,'HC - an - pca0',hc)
plot_mean_ts(axs[1,2],t.cface_mean_ts_an_pca0,'SZA - an - pca0',sza)
plot_mean_ts(axs[2,2],t.cface_mean_ts_an_pca0,'SZ - an - pca0',sz)
fig.suptitle('cface_mean_ts_pca0')
fig.tight_layout()
plt.show()