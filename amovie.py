"""
Analyse movieDI face data for Ricky (after OpenFace .csv files have been generated)
Mainly using the detailed.csv log file, and the OpenFace-processed .csv file with action unit time series
Resample action unit series from being indexed by frames as in the OpenFace .csv, to be indexed by time (sec) relative to the movie
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from acommonvars import *
import acommonfuncs
from glob import glob
from scipy.interpolate import interp1d

### SETTABLE PARAMETERS ###

static_or_dynamic = 'static' #whether au_static flag was used in OpenFace execution or not (default 'static', alternatively 'dynamic')
min_success = 0.95 #minimum proportion of successful webcam frames for a subject to be included. Default 0.95
target_fps=20 #target framerate to resample time series to

movie_actual_start_time_sec = 2 #2sec for non MRI version, 10 sec for MRI version
movie_actual_duration_sec = 253 #253sec for Ricky stimulus, ??? for DISFA stimulus

gap_sec = 0 #use webcam outputs from this many seconds after movie starts, until this many seconds before movie ends. Default 0. Could set to 0.5 to avoid the first 0.5sec of the movie, and the last 0.5sec of the movie
start_time_sec = 2 + gap_sec #actual start time used for analysis
duration_sec = movie_actual_duration_sec - gap_sec

HC=((healthy_attended_inc) & (t.valid_movieo==1)) #healthy group
PT=((clinical_attended_inc) & (t.valid_movieo==1)) #patient group
SZ = ((sz_attended_inc) & (t.valid_movieo==1)) #schizophrenia subgroup
SZA = ((sza_attended_inc) & (t.valid_movieo==1)) #schizoaffective subgroup
HC,PT,SZ,SZA = subs[HC],subs[PT],subs[SZ],subs[SZA]

### DO THE ANALYSIS ###
subject = SZ[0] #pick a subject. Ideally we would loop over a group of subjects

all_frames,aus = acommonfuncs.get_openface_table('movieDI',subject,static_or_dynamic,min_success=min_success) #Get the OpenFace output .csv for this subject
detailed = acommonfuncs.get_beh_data('movieDI',subject,'detailed',use_MRI_task=False) #Get detailed webcam frametimes from *detailed.csv

#Quality checks using the *summary.csv
summary = acommonfuncs.get_beh_data('movieDI',subject,'summary',use_MRI_task=False) #Get summary data from *summary.csv
if summary is not None:
    summary = {i[0]:i[1] for i in summary.values} #convert face summary array into dictionary
    assert(np.abs(summary['movietimestart'] - movie_actual_start_time_sec) < 0.5) #ensure movie started close to when it should have
    assert(np.abs(summary['actualmovietime'] - movie_actual_duration_sec) < 0.5)

#Use *detailed.csv to get estimated timestamp for each webcam frame
times_eachframe = np.zeros(len(all_frames),dtype=float) #holds the estimated timestamp for each webcam frame
times_eachframe[:]=np.nan
for index in range(aus.shape[0]): #for each row of OpenFace output .csv (corresponding to webcam frames)
    framenum = index + 1 
    try: #if that webcam frame coincides with any movie frames (appears in data.ptframenums)
        times_eachframe[index]=detailed.fliptimes[detailed.ptframenums==framenum].iloc[0] #find the timestamp of the first loop iteration which corresponds to that webcam frame
    except:
        pass

#Resample action unit time series at 20fps with linear interpolation
interp_aus=interp1d(times_eachframe,aus,kind='linear',axis=0,fill_value='extrapolate') 
times_regular = np.arange(start_time_sec,duration_sec,1/target_fps)
aust=interp_aus(times_regular)
aust=pd.DataFrame(aust)
aust.columns = aus.columns #use aust for any downstream analyses

#Plot AU12 time series, for a single participant
fig,ax=plt.subplots()
ax.plot(times_regular,aust['AU12'],color='blue',label='x')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_xlabel('Time (s)')
ax.legend()
plt.show()
