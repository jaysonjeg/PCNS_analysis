"""
Analyse movieDI face data for Ricky (after OpenFace .csv files have been generated)
Mainly using the detailed.csv log file, and the OpenFace-processed .csv file with action unit time series
Resample action unit series from being indexed by frames as in the OpenFace .csv, to be indexed by time (sec) relative to the movie
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.interpolate import interp1d
from acommonvars import *
import acommonfuncs


### SETTABLE PARAMETERS ###

min_success = 0.95 #minimum proportion of successful webcam frames for a subject to be included. Default 0.95
target_fps=20 #target framerate to resample time series to
movie_actual_start_time_sec = 2 #2sec for non MRI version, 10 sec for MRI version
movie_actual_duration_sec = 253 #253sec for Ricky stimulus, ??? for DISFA stimulus
gap_sec = 0 #use webcam outputs from this many seconds after movie starts, until this many seconds before movie ends. Default 0. Could set to 0.5 to avoid the first 0.5sec of the movie, and the last 0.5sec of the movie
start_time_sec = 2 + gap_sec #actual start time used for analysis
duration_sec = movie_actual_duration_sec - gap_sec
times_regular = np.arange(start_time_sec,duration_sec,1/target_fps)

c = acommonfuncs.clock()



def get_resampled_time_series(subject,static_or_dynamic,r_or_c):
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

if __name__=='__main__':

    

    group = 'group02' #the grouping variable
    load_table=True
    outliers = []
    
    #new_columns = ['use_ricky','ricky_outliers']
    if load_table:
        t = acommonfuncs.add_table(t,'outcomes_ricky.csv')
        #t = acommonfuncs.str_columns_to_literals(t,['sinus_ts'])
    else:
        new_columns = ['use_ricky','ricky_outliers','ricky_aussr','ricky_ausdc']
        t=acommonfuncs.add_columns(t,new_columns[2:])
        t['use_ricky'] = ((include) & (t.valid_movieo==1)) 
        t['ricky_outliers'] = t.subject.isin(outliers)
        #t=acommonfuncs.add_columns(t,['sinus_ts'])
        for t_index in range(len(t)):
            subject=t.subject[t_index]
            if (t['use_ricky'][t_index]) and (t.subject[t_index] not in outliers):
                print(f'{c.time()[1]}: Subject {subject}')
                aussr = get_resampled_time_series(subject,'static','r')
                t.at[t_index,'ricky_aussr'] = list([list(i) for i in aussr.values])
                ausdc = get_resampled_time_series(subject,'dynamic','c')
                t.at[t_index,'ricky_ausdc'] = list([list(i) for i in ausdc.values])
                
            if subject=='009': 
                print(f'stopped at subject {subject}')
                #break

        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_ricky.csv')

    x=t.ricky_aussr[2]
    print(type(x))
    #assert(0)


    for i in range(t.shape[0]):
        element=t.at[i,'ricky_aussr']
        if type(element)==str:
            print(c.time()[1])
            x=eval(element)


    print(c.time()[1])
    for i in range(t.shape[0]): 
        element = t.at[i,'ricky_aussr']
        if type(element)==str:
            new_element = np.vstack(eval(element)).astype(np.float32)
            #new_element = pd.DataFrame(new_element)
            #new_element.columns = aus_labels
            t.at[i,'ricky_aussr'] = new_element
    print(c.time()[1])
    #as above but for dynamic
    for i in range(t.shape[0]):
        element = t.at[i,'ricky_ausdc']
        if type(element)==str:
            new_element = np.vstack(eval(element)).astype(np.float32)
            t.at[i,'ricky_ausdc'] = new_element
    print(c.time()[1])

    x=t.ricky_aussr[2]
    print(type(x))
    #assert(0)

    valid_indices_bool = (t.use_ricky) & (~t.ricky_outliers)
    vib = valid_indices_bool
    valid_indices = np.where(vib)[0]

    print(c.time()[1])
    #concatenate 2D arrays in ricky_aussr into a 3D array
    aussr = np.zeros((len(valid_indices),len(times_regular),len(aus_labels)),dtype=np.float32)
    for i in range(valid_indices_bool.sum()):
        aussr[i,:,:] = t.ricky_aussr[valid_indices[i]]
    print(c.time()[1])
    #as above, but for dynamic
    ausdc = np.zeros((len(valid_indices),len(times_regular),len(aus_labels)),dtype=np.float32)
    for i in range(valid_indices_bool.sum()):
        ausdc[i,:,:] = t.ricky_ausdc[valid_indices[i]] 
    print(c.time()[1])

    

    tgroup = t[group][vib]
    #tgroup = t[group][30:30+92]
    #print('USING RANDOM GROUPS OF HC AND CC')
    #aussr = np.random.normal(size=(92,5020,16))
    #ausdc = np.random.normal(size=(92,5020,16))


    AU12 = np.where(np.array(aus_labels)=='AU12')[0][0]
    aussr_12 = aussr[:,:,AU12]
    aussr_12 = pd.DataFrame(aussr_12)
    aussr_12['subject'] = list(range(0,len(aussr_12)))
    aussr_12[group] = pd.Series(tgroup.values, dtype = 'str')

    print(c.time()[1])
    assert(0)
    x = pd.melt(aussr_12, id_vars = ['subject',group], value_vars = list(range(len(times_regular))), var_name = 'time',value_name = 'AU12')
    x['time'] = pd.Series(times_regular[list(x.time.values)],dtype=np.float32)
    fig,ax=plt.subplots()
    print(c.time()[1])
    sns.lineplot(data=x, x="time", y="AU12", ax=ax,hue=group,palette=colors, n_boot=50)
    print(c.time()[1])
    plt.show()
    assert(0)

    from scipy import stats
    aussr_mad = stats.median_abs_deviation(aussr,axis=1)
    aussr_mean = aussr.mean(axis=1)
    aussr_std = aussr.std(axis=1)


    gps = list(tgroup.unique())
    if '' in gps: gps.remove('')

    def get_AU_df(aus,action_unit):
        nAU = np.where(np.array(aus_labels)==action_unit)[0][0]
        array2D = aus[:,:,nAU]
        def apply(func):
            return pd.Series(func(array2D,axis=1),dtype=np.float32)
        strings = [f'{action_unit}_{i}' for i in ['mean','std','median','mad']]
        series = [apply(i) for i in [np.mean,np.std,np.median,stats.median_abs_deviation]]
        dictionary = {key:value for key,value in zip(strings,series)}
        dictionary[group] = pd.Series(tgroup.values , dtype = 'str')
        return pd.DataFrame(dictionary)

    def compare(aus,outcome,title=''):
        fig,axs=plt.subplots(4,4,figsize=(12,8))
        for i in range(len(aus_labels)):
            action_unit = aus_labels[i]
            string = f'{action_unit}_{outcome}'
            ax = axs[np.unravel_index(i,(4,4))]
            df = get_AU_df(aus,action_unit)
            sns.stripplot(ax=ax,data = df, x=group, hue=group,palette=colors,y=string,legend=False)
            group1 = df.loc[df[group]==gps[0],string]
            group2 = df.loc[df[group]==gps[1],string]
            diff = group1.mean() - group2.mean()
            pval = stats.ttest_ind(group1,group2).pvalue
            ax.set_title(f'diff {diff:.2} p={pval:.2f}')
        fig.suptitle(title)
        fig.tight_layout()

    #Scatterplots
    acommonfuncs.pairplot(get_AU_df(aussr,'AU12'),vars=['mean','std'],height=1.5,robust=True,group=group)
    acommonfuncs.pairplot(get_AU_df(aussr,'AU12'),vars=['std','mad'],height=1.5,robust=True,group=group)

    #Compare across groups
    compare(aussr,'mean','sr')
    compare(aussr,'std','sr')
    compare(ausdc,'std','dc')



    plt.show(block=False)