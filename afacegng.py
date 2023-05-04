"""
Analyse facegng data
Adapted from aff1.py
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re

def facegng_get_subject_data(subject,show_plots):
    contents=glob(f"{top_folder}\\PCNS_{subject}_BL\\beh\\facegng*\\") #find the FF1 folder for this subject
    assert(len(contents)==1) 
    resultsFolder=contents[0]
    df=pd.read_csv(glob(f"{resultsFolder}*out.csv")[0]) # make log csv into dataframe
    emots=df['emot'] 
    RT=df['RT']
    
    RT_fear=RT[emots=='fear']
    RT_calm=RT[emots=='calm']
    RT_FearMinusCalm = RT_fear.mean() - RT_calm.mean() #Difference in reaction time between fear and calm conditions, mean for the subject

    if show_plots: #Plot for particular subject
        pass
    
    return RT_FearMinusCalm

top_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw\\"
files_with_task=glob(f"{top_folder}\\PCNS_*_BL\\beh\\facegng*\\")
subjects=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task
subjects_to_exclude=['020'] #exclude these subjects
"""
020: no ppg

"""


subjects_with_task = [subject for subject in subjects if subject not in subjects_to_exclude]

temp=[facegng_get_subject_data(subject,show_plots=False) for subject in subjects_with_task]
RTs_FearMinusCalm = np.array(temp).T.tolist() #convert into separate lists

from scipy import stats
x=stats.ttest_1samp(RTs_FearMinusCalm,popmean=0).pvalue/2

print(x)