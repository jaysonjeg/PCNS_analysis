"""
Analyse movieDI data
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re
from acommon import *

def ff1_get_subject_data(subject,show_plots):
    contents=glob(f"{top_folder}\\PCNS_{subject}_BL\\beh\\FF1*\\") #find the FF1 folder for this subject
    print(subject)
    assert(len(contents)==1) 
    resultsFolder=contents[0]
    df=pd.read_csv(glob(f"{resultsFolder}*out.csv")[0]) # make log csv into dataframe
    emots=df['emot'] 
    falsefeedback=(df['stim_FBmultiplier']!=1)
    ratings=df['rating']   

    NE_false=((emots=='NE') & falsefeedback) #for this subject, get trial numbers which were Neutral and false feedback
    NE_true=((emots=='NE') & ~falsefeedback)
    ratings_NE_false=ratings[NE_false] #get ratings corresponding to these trial numbers
    ratings_NE_true=ratings[NE_true]
    ratings_NE=ratings[emots=='NE']
    ratings_AN=ratings[emots=='AN']
    ratings_HA=ratings[emots=='HA']

    print(ratings_NE_false.shape)
    print(ratings_NE_true.shape)

    ntrialsWithAbsentRatings=sum(np.isnan(ratings)) #no of trials with absent ratings
    ANminusNE=ratings_AN.mean() - ratings_NE.mean() #Anger minus Neutral ratings, mean for the subject
    FalseMinusTrue=ratings_NE_false.mean() - ratings_NE_true.mean()   #Difference between false feedback and true feedback, mean for the subject

    if show_plots: #Plot for particular subject
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(f"FF1 sub {subject}, absent {ntrialsWithAbsentRatings}\n ANminusNE {ANminusNE:.2f} FminusT {FalseMinusTrue:.2f}")
        axis=axs[0]
        axis.hist([ratings_NE,ratings_HA,ratings_AN],label=['NE','HA','AN'])
        axis.legend()
        axis.set_xlabel('Emotional intensity rating')
        axis.set_title('')
        axis=axs[1]
        axis.hist([ratings_NE_false,ratings_NE_true],label=['NE_falsefeedback','NE_truefeedback'])
        axis.legend()
        axis.set_xlabel('Emotional intensity rating')
        axis.set_title('')
        plt.show()
    
    return ntrialsWithAbsentRatings, ANminusNE, FalseMinusTrue


task='FF1'
top_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw\\"
files_with_task=glob(f"{top_folder}\\PCNS_*_BL\\beh\\{task}*\\")
subjects=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task

'''
subjects_to_exclude=['020'] #exclude these subjects, e.g. ['020']
"""
020: no ppg

"""
'''
subjects_to_exclude=[]

subjects_with_task = [subject for subject in subjects if subject not in subjects_to_exclude]

temp=[ff1_get_subject_data(subject,show_plots=False) for subject in subjects_with_task]
[ntrialsWithAbsentRatings, ANminusNE, FalseMinusTrue] = np.array(temp).T.tolist() #convert into separate lists

#Plot summary for all participants
fig, axs = plt.subplots(nrows=1, ncols=3)
fig.suptitle('All participants')
axis=axs[0]
axis.hist(ntrialsWithAbsentRatings)
axis.set_title('ntrialsWithAbsentRatings')
axis=axs[1]
axis.hist(ANminusNE)
axis.set_title('ANminusNE')
axis=axs[2]
axis.hist(FalseMinusTrue)
axis.set_title('FalseMinusTrue')
plt.show()

from scipy import stats
x=stats.ttest_1samp(FalseMinusTrue,popmean=0).pvalue/2