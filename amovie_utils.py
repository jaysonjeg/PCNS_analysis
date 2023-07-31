import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import acommonfuncs


def get_resampled_time_series(subject,static_or_dynamic,r_or_c,times_regular):

    ### Same parameters in amovie.py
    min_success = 0.95 #minimum proportion of successful webcam frames for a subject to be included. Default 0.95
    movie_actual_start_time_sec = 2 #2sec for non MRI version, 10 sec for MRI version
    movie_actual_duration_sec = 253 #253sec for Ricky stimulus, ??? for DISFA stimulus

    #### Get the data
    all_frames,aus_raw,success = acommonfuncs.get_openface_table('movieDI',subject,static_or_dynamic,r_or_c) #Get the OpenFace output .csv for this subject. all_frames is just all integers from 1 to total number of frames. aus is dataframe of nwebcamframes * nAUs. success is a boolean array of length nwebcamframes, indicating whether OpenFace was able to detect a face in that frame
    detailed = acommonfuncs.get_beh_data('movieDI',subject,'detailed',use_MRI_task=False) #Get detailed webcam frametimes from *detailed.csv
    try: #a few subjects don't have a summary file
        summary = acommonfuncs.get_beh_data('movieDI',subject,'summary',use_MRI_task=False,header=None) #Get summary data from *summary.csv
    except:
        summary = None

    #### Quality checks
    assert(success.mean() > min_success) #ensure that the webcam was working most of the time
    #Quality checks using the *summary.csv
    if summary is not None:
        summary = {i[0]:i[1] for i in summary.values} #convert face summary array into dictionary
        assert(np.abs(summary['movietimestart'] - movie_actual_start_time_sec) < 0.5) #ensure movie started close to when it should have
        assert(np.abs(summary['actualmovietime'] - movie_actual_duration_sec) < 2)
   
    #### Use *detailed.csv to get estimated timestamp for each webcam frame (slow)
    times_eachframe = np.zeros(len(all_frames),dtype=float) #holds the estimated timestamp for each webcam frame
    times_eachframe[:]=np.nan
    for index in range(aus_raw.shape[0]): #for each row of OpenFace output .csv (corresponding to webcam frames)
        framenum = index + 1 
        try: #if that webcam frame coincides with any movie frames (appears in data.ptframenums)
            times_eachframe[index]=detailed.fliptimes[detailed.ptframenums==framenum].iloc[0] #find the timestamp of the first loop iteration which corresponds to that webcam frame
        except:
            pass
    
    #### Resample action unit time series at 20fps with linear interpolation
    interp_aus=interp1d(times_eachframe,aus_raw,kind='linear',axis=0,fill_value='extrapolate') 
    aus=interp_aus(times_regular).astype(np.float32)
    aus=pd.DataFrame(aus)
    aus.columns = aus.columns #use aus for any downstream analyses
    return aus


def get_all_resampled_time_series(static_or_dynamic,r_or_c,t,valid,times_regular,aus_labels,c):
    #Get resampled time series for all subjects and put into 3D array
    all_aus = np.zeros((len(valid),len(times_regular),len(aus_labels)),dtype=np.float32)
    for i in range(len(valid)):
        subject = t.subject[valid[i]]
        print(f'{c.time()[1]}: Subject {subject}')
        all_aus[i,:,:] = get_resampled_time_series(subject,static_or_dynamic,r_or_c,times_regular).values
    return all_aus

def plot_sample_time_series(action_unit,aus_labels,times_regular,aussr,ausdr,ausdc,valid,t):
    nAU = np.where(np.array(aus_labels)==action_unit)[0][0]
    ids = [0,5,10,15,20,25,30,35,40]
    fig,axs=plt.subplots(3,3)
    for i in range(9):
        ax = axs[np.unravel_index(i,(3,3))]
        ax.plot(times_regular,aussr[ids[i],:,nAU],color='b',alpha=0.4)
        ax.plot(times_regular,ausdr[ids[i],:,nAU],color='r',alpha=0.4)
        ax.plot(times_regular,ausdc[ids[i],:,nAU],color='darkred',alpha=0.4)
        ax.set_title(f'{t.subject[valid[ids[i]]]}')
        ax.set_xlabel('time')
        ax.set_ylabel(action_unit)
    fig.tight_layout()

def get_aus_df_long(aus,action_unit,aus_labels,group,times_regular,tgroup):
    #Given 3Darray of action unit time series for all subjects, return data for just one AU in long format with columns for subject, group, time and AU intensity
    nAU = np.where(np.array(aus_labels)==action_unit)[0][0]
    aus_AU = aus[:,:,nAU]
    aus_AU = pd.DataFrame(aus_AU)
    aus_AU['subject'] = list(range(0,len(aus_AU)))
    aus_AU[group] = pd.Series(tgroup.values, dtype = 'str')
    x = pd.melt(aus_AU, id_vars = ['subject',group], value_vars = list(range(len(times_regular))), var_name = 'time',value_name = action_unit)
    x['time'] = pd.Series(times_regular[list(x.time.values)],dtype=np.float32)
    return x

def get_AU_df(aus,action_unit,aus_labels,tgroup,group):
    nAU = np.where(np.array(aus_labels)==action_unit)[0][0]
    array2D = aus[:,:,nAU]
    def apply(func):
        return pd.Series(func(array2D,axis=1),dtype=np.float32)
    strings = [f'{action_unit}_{i}' for i in ['mean','std','median','mad','cv']]
    series = [apply(i) for i in [np.mean,np.std,np.median,stats.median_abs_deviation,stats.variation]]
    dictionary = {key:value for key,value in zip(strings,series)}
    dictionary[group] = pd.Series(tgroup.values , dtype = 'str')
    return pd.DataFrame(dictionary)

def compare(aus,outcome,aus_labels,gps,tgroup,group,colors,title=''):
    fig,axs=plt.subplots(4,4,figsize=(12,8))
    for i in range(len(aus_labels)):
        action_unit = aus_labels[i]
        string = f'{action_unit}_{outcome}'
        ax = axs[np.unravel_index(i,(4,4))]
        df = get_AU_df(aus,action_unit,aus_labels,tgroup,group)
        sns.stripplot(ax=ax,data = df, x=group, hue=group,palette=colors,y=string,legend=False)
        group1 = df.loc[df[group]==gps[0],string].dropna()
        group2 = df.loc[df[group]==gps[1],string].dropna()
        diff = group1.mean() - group2.mean()
        diff_median = np.median(group1) - np.median(group2)
        p_ttest = stats.ttest_ind(group1,group2).pvalue
        p_MW = stats.mannwhitneyu(group1,group2).pvalue
        ax.set_title(f'Dmean {diff:.2} ttest p={p_ttest:.2f} Dmed {diff_median:.2f} MW p={p_MW:.2f}')
    fig.suptitle(title)
    fig.tight_layout()