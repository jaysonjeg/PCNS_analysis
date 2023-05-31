"""
Analyse proprio data
Adapted from aFF1.py
Kind of copied from Analysis_proprio.m
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re, scipy.io, acommonvars

AU_to_plot = 'AU12'
top_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw\\"
files_with_task=glob(f"{top_folder}\\PCNS_*_BL\\beh\\proprio*\\")
subjects=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task
subjects_to_exclude=['018'] #exclude these subjects, e.g. ['020']
"""
018 has zero values for AU12 all the time
"""
subjects_with_task = [subject for subject in subjects if subject not in subjects_to_exclude]

t=acommonvars.get_redcap()
group_numbers = [t.group[t.record_id==int(subject)].iloc[0] for subject in subjects]
group = [{1:'healthy',2:'clinical'}[i] for i in group_numbers]


def get_proprio_data(subject,AU_to_plot='AU12'):
    aulabels_list=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45'] 
    nAU = aulabels_list.index(AU_to_plot)

    contents=glob(f"{top_folder}\\PCNS_{subject}_BL\\beh\\proprio*\\") #find the FF1 folder for this subject
    assert(len(contents)==1) 
    resultsFolder=contents[0]
    mat=scipy.io.loadmat(glob(f"{resultsFolder}*.mat")[0])
    data=mat['data'] #4D: blocks x trials x (stimface, ptface) x AU intensities for a single frame from OpenFace
    delays=mat['delays'] #array(nblocks,ntrials)
    data2 = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2],data.shape[3])) #collapse blocks and trials into one dimension

    stimface = data2[:,0,nAU]
    ptface = data2[:,1,nAU]
    return stimface,ptface

fig,axes=plt.subplots(nrows=5,ncols=5,figsize=(12,30))
fig.suptitle('H Healthy, C Clinical, regression slope')
for i in range(len(subjects_with_task)):
    stimface,ptface=get_proprio_data(subjects_with_task[i],AU_to_plot) 
    coeffs=np.polyfit(stimface,ptface,1)
    predict_function = np.poly1d(coeffs)
    
    axis=axes[np.unravel_index(i,axes.shape)]
    axis.plot(stimface,ptface,'b.',stimface,predict_function(stimface),'--k')
    axis.set_title(f'{group[i][0].capitalize()}: {coeffs[0]:.3f}')
    axis.set_xlim([0,4])
    axis.set_ylim([0,4])
    
plt.show()