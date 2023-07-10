"""
Analyse Facial mirring task (sinus)
10 trials, each lasting 20s

Webcam recording received at 30fps, but task saved data (and stimulus display) are at 25 fps (0.04 sec/frame)

First 2 trials are no feedback, predictable
Next 5 trials are no feedback, pjump=[0.025,0.03,0.035,0.04,0.045]
Last 3 trials are feedback present, predictable

New columns
    use_sinus
    sinus_outliers
"""

import numpy as np, pandas as pd, seaborn as sns, pingouin as pg
import matplotlib.pyplot as plt
from glob import glob
import re, scipy.io
import warnings
from acommonvars import *
import acommonfuncs


def get_sinus_data(subject):
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\sinus*\\") #find the FF1 folder for this subject
    assert(len(contents)==1) 
    resultsFolder=contents[0]
    mat=scipy.io.loadmat(glob(f"{resultsFolder}*.mat")[0])
    data=mat['data'] #4D: blocks x trials x (stimface, ptface) x AU intensities for a single frame from OpenFace
    delays=mat['delays'] #array(nblocks,ntrials)
    data2 = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2],data.shape[3])) #collapse blocks and trials into one dimension

AU_to_plot = 'AU12'
aulabels_list=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45'] 
nAU = aulabels_list.index(AU_to_plot)


ntrials = 10 #number of trials per subject
nframes = 500 #number of frames per trial
time_interval = 0.04 #seconds per frame

nseconds_per_trial = nframes*time_interval
times_trial_regular = np.linspace(time_interval, nseconds_per_trial, nframes)

outliers=[] #exclude these subjects

if __name__=='__main__':
    c = acommonfuncs.clock()

    ### SETTABLE PARAMETERS 
    group = 'group02' #the grouping variable
    load_table=False

    ### Get data
    new_columns = ['use_sinus','sinus_outliers','sinus_ts']

    if load_table:
        t = acommonfuncs.add_table(t,'outcomes_sinus.csv')
        t = acommonfuncs.str_columns_to_literals(t,['sinus_ts'])
    else:
        t['use_sinus'] = ((include) & (t.valid_sinuso==1)) 
        t['prop_outliers'] = t.subject.isin(outliers)
        t=acommonfuncs.add_columns(t,['sinus_ts'])
        for i in range(len(t)):
            if t['use_sinus'][i]:
                subject=t.subject[i]
                print(f'{c.time()[1]}: Subject {subject}')



                contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\sinus*\\") #find the FF1 folder for this subject
                assert(len(contents)==1) 
                resultsFolder=contents[0]
                mat=scipy.io.loadmat(glob(f"{resultsFolder}*.mat")[0])
                ausdata = mat['ausdata'].squeeze() #10 trials * 500 timepoints, with each element being (n,17) array of AU intensities. n is usually 1, but can be 2 (if 2 samples received from webcam) or 0

                f = lambda x: x[0,nAU]
                ftype = lambda x: type(x)
                flen = lambda x: len(x)
                x=np.vectorize(flen)(ausdata)

                fmean = lambda x: list(x.mean(axis=0))

                #ausdata[7,3], ausdata[0,3]

                #In each frame, find the mean (if there are more than 1 webcame frames)
                for i in range(ausdata.shape[0]):
                    for j in range(ausdata.shape[1]):
                        if len(ausdata[i,j])!=0:
                            ausdata[i,j] = ausdata[i,j].mean(axis=0)
                        else:
                            temp = np.array([np.nan]*len(aulabels_list))
                            ausdata[i,j]=temp


                temp = []
                for trial in range(ntrials):
                    ausdata_trial = np.vstack(ausdata[trial,:])
                    temp.append(pd.DataFrame(ausdata_trial).interpolate(method='linear',axis=0).values)
                aus = np.transpose(np.dstack(temp) , axes=[2,1,0])

                assert(0)
                """
                sinus_ts=get_sinus_data(subject) 
                t.at[i,'prop_stim'] = list(stimface)
                t.at[i,'prop_resp'] = list(respface)

                t.at[i,'prop_stim_min'] = min(stimface)
                t.at[i,'prop_stim_max'] = max(stimface)
                t.at[i,'prop_stim_range'] = t.prop_stim_max[i] - t.prop_stim_min[i]

                if t.subject[i] not in outliers:
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stimface,respface)
                    t.at[i,'prop_slope'] = slope
                    t.at[i,'prop_intercept'] = intercept
                    t.at[i,'prop_r2'] = r_value**2
                """

        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_sinus.csv')
