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
    summary = acommonfuncs.get_beh_data('movieDI',subject,'summary',use_MRI_task=False,header=None) #Get summary data from *summary.csv

    #### Quality checks
    assert(success.mean() > min_success) #ensure that the webcam was working most of the time
    #Quality checks using the *summary.csv
    if summary is not None:
        summary = {i[0]:i[1] for i in summary.values} #convert face summary array into dictionary
        assert(np.abs(summary['movietimestart'] - movie_actual_start_time_sec) < 0.5) #ensure movie started close to when it should have
        assert(np.abs(summary['actualmovietime'] - movie_actual_duration_sec) < 0.5)
   
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
    load_table=False
    outliers = []

    #new_columns = ['use_ricky','ricky_outliers']
    if load_table:
        t = acommonfuncs.add_table(t,'outcomes_ricky.csv')
        #t = acommonfuncs.str_columns_to_literals(t,['sinus_ts'])
    else:
        t=acommonfuncs.add_columns(t,['ricky_aussr'])
        t['use_ricky'] = ((include) & (t.valid_movieo==1)) 
        t['ricky_outliers'] = t.subject.isin(outliers)
        #t=acommonfuncs.add_columns(t,['sinus_ts'])
        for t_index in range(len(t)):
            if (t['use_ricky'][t_index]) and (t.subject[t_index] not in outliers):
                subject=t.subject[t_index]
                print(f'{c.time()[1]}: Subject {subject}')
                aussr = get_resampled_time_series(subject,'static','r')

                t.at[t_index,'ricky_aussr'] = aussr
                if subject=='009': break

        #t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_ricky.csv')


    valid_indices = (t.use_ricky) & (~t.ricky_outliers)
    valid_indices = t.subject.isin(['005','006','007'])
    #t.loc[valid_indices,'ricky_aussr']

    #apply valid_indices to original dataframe t to get a reduced version.
    #apply valid_indices to t filled with aussr, then convert into a 3D array of aussr's. 