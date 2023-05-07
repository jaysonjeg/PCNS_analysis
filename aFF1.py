"""
Analyse movieDI data
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re
from acommon import *
from scipy import stats

def get_data_table(subject):
    print(subject)
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\FF1*\\") #find the FF1 folder for this subject
    assert(len(contents)==1) 
    resultsFolder=contents[0]
    df=pd.read_csv(glob(f"{resultsFolder}*out.csv")[0]) # make log csv into dataframe
    emot=df['emot'].values 
    falsefeedback=(df['stim_FBmultiplier']!=1).values
    rating=df['rating'].values   
    return emot,falsefeedback,rating

def contains_all_responses(rating):
    #Return True if rating contains each of 0, 1, 2, 3
    return all([i in rating for i in [0,1,2,3]])

def ff1_get_subject_data(subject,show_plots):
    emots,falsefeedback,ratings =  get_data_table(subject)

    NE_false=((emots=='NE') & falsefeedback) #for this subject, get trial numbers which were Neutral and false feedback
    NE_true=((emots=='NE') & ~falsefeedback)
    ratings_NE_false=ratings[NE_false] #get ratings corresponding to these trial numbers
    ratings_NE_true=ratings[NE_true]
    ratings_NE=ratings[emots=='NE']
    ratings_AN=ratings[emots=='AN']
    ratings_HA=ratings[emots=='HA']

    ntrialsWithAbsentRatings=sum(np.isnan(ratings)) #no of trials with absent ratings
    ANminusNE=ratings_AN.mean() - ratings_NE.mean() #Anger minus Neutral ratings, mean for the subject
    NE_FalseMinusTrue=ratings_NE_false.mean() - ratings_NE_true.mean()   #Difference between false feedback and true feedback, mean for the subject

    if show_plots: #Plot for particular subject
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(f"FF1 sub {subject}, absent {ntrialsWithAbsentRatings}\n ANminusNE {ANminusNE:.2f} FminusT {NE_FalseMinusTrue:.2f}")
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
    
    return ntrialsWithAbsentRatings, ANminusNE, NE_FalseMinusTrue


task='FF1'

files_with_task=glob(f"{data_folder}\\PCNS_*_BL\\beh\\{task}*\\")
subjects_with_task=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task

HC=((healthy_didmri_inc) & (t.valid_ffi==1)) #healthy group
PT=((clinical_didmri_inc) & (t.valid_ffi==1)) #patient group
SZ = sz_didmri_inc #schizophrenia subgroup
SZA = sza_didmri_inc #schizoaffective subgroup
HC,PT,SZ,SZA = subs[HC],subs[PT],subs[SZ],subs[SZA]

r = [get_data_table(i) for i in HC[0:5]] #r is a list of tuples of length 3. 
emots,falsefeedbacks,ratings=zip(*r) #Split r into 3 tuples of arrays
ntrials=len(emots[0])

NE = [i=='NE' for i in emots]
AN = [i=='AN' for i in emots]
HA = [i=='HA' for i in emots]

NE_false = [j & k for j,k in zip(NE,falsefeedbacks)] #trials corresponding to Neutral trials with false feedback
NE_true = [j & ~k for j,k in zip(NE,falsefeedbacks)] #trials corresponding to Neutral trials with true feedback
AN_false = [j & k for j,k in zip(AN,falsefeedbacks)] #trials corresponding to Anger trials with false feedback
AN_true = [j & ~k for j,k in zip(AN,falsefeedbacks)] #trials corresponding to Anger trials with true feedback
HA_false = [j & k for j,k in zip(HA,falsefeedbacks)] #trials corresponding to Happy trials with false feedback
HA_true = [j & ~k for j,k in zip(HA,falsefeedbacks)] #trials corresponding to Happy trials with true feedback

ratings_NE = [i[j] for i,j in zip(ratings,NE)] #get ratings corresponding to Neutral trials
ratings_NE_false = [i[j] for i,j in zip(ratings,NE_false)] #get ratings corresponding to Neutral trials with false feedback
ratings_NE_true = [i[j] for i,j in zip(ratings,NE_true)] #get ratings corresponding to Neutral trials with true feedback
ratings_AN = [i[j] for i,j in zip(ratings,AN)] #get ratings corresponding to Anger trials
ratings_AN_false = [i[j] for i,j in zip(ratings,AN_false)] #get ratings corresponding to Anger trials with false feedback
ratings_AN_true = [i[j] for i,j in zip(ratings,AN_true)] #get ratings corresponding to Anger trials with true feedback
ratings_HA = [i[j] for i,j in zip(ratings,HA)] #get ratings corresponding to Happy trials
ratings_HA_false = [i[j] for i,j in zip(ratings,HA_false)] #get ratings corresponding to Happy trials with false feedback
ratings_HA_true = [i[j] for i,j in zip(ratings,HA_true)] #get ratings corresponding to Happy trials with true feedback


### Quality check of behavioural data ###
quality_all_possible_responses = [contains_all_responses(i) for i in ratings]

perc_trials_with_absent_ratings = [100*sum(np.isnan(i))/ntrials for i in ratings] #% of trials with absent ratings
quality_absent_ratings = [i<10 for i in perc_trials_with_absent_ratings] #whether % of trials with absent ratings is less than 10%

ANminusNE = [np.nanmean(i)-np.nanmean(j) for i,j in zip(ratings_AN,ratings_NE)] #Anger minus Neutral ratings, mean for the subject
quality_ANminusNE = [i>0 for i in ANminusNE] #whether ANminusNE is positive

#Setwise intersection of quality checks
quality = [i & j & k for i,j,k in zip(quality_all_possible_responses,quality_absent_ratings,quality_ANminusNE)] #whether all quality checks are passed


### The more interesting behavioural outputs ###
NE_FalseMinusTrue = [np.nanmean(i)-np.nanmean(j) for i,j in zip(ratings_NE_false,ratings_NE_true)] #False feedback ratings minus true feedback ratings, for NE, mean for the subject
AN_FalseMinusTrue = [np.nanmean(i)-np.nanmean(j) for i,j in zip(ratings_AN_false,ratings_AN_true)] #False feedback ratings minus true feedback ratings, for AN, mean for the subject
HA_FalseMinusTrue = [np.nanmean(i)-np.nanmean(j) for i,j in zip(ratings_HA_false,ratings_HA_true)] #False feedback ratings minus true feedback ratings, for HA, mean for the subject
sum_FalseMinusTrue = [i+j+k for i,j,k in zip(NE_FalseMinusTrue,AN_FalseMinusTrue,HA_FalseMinusTrue)]

#Plot quality summary for all participants
fig, axs = plt.subplots(nrows=1, ncols=3)
fig.suptitle('Distribution of beh quality checks')
def plotfunc2(axis,data,title):
    axis.hist(data)
    axis.set_title(title)
plotfunc2(axs[0],perc_trials_with_absent_ratings,'perc_trials_with_absent_ratings')
plotfunc2(axs[1],ANminusNE,'ANminusNE')
plotfunc2(axs[2],[int(i) for i in quality_all_possible_responses],'has_all_possible_responses')
plt.show()

# NE_false_minus_true_pass is a subset of NE_false_minus_true, where quality checks are passed
NE_FalseMinusTrueq = [i for i,j in zip(NE_FalseMinusTrue,quality) if j]
AN_FalseMinusTrueq = [i for i,j in zip(AN_FalseMinusTrue,quality) if j]
HA_FalseMinusTrueq = [i for i,j in zip(HA_FalseMinusTrue,quality) if j]
sum_FalseMinusTrueq = [i for i,j in zip(sum_FalseMinusTrue,quality) if j]

fig, axs = plt.subplots(nrows=1, ncols=3)
fig.suptitle('Distribution of False ratings minus True ratings')
def plotfunc1(axis,data,title):
    axis.hist(data)
    p=stats.ttest_1samp(data,popmean=0).pvalue/2
    axis.set_title(f'{title}: mean={np.nanmean(data):.2f}\n p={p:.2f}')
plotfunc1(axs[0],NE_FalseMinusTrueq,'NE_FalseMinusTrueq')
plotfunc1(axs[1],AN_FalseMinusTrueq,'AN_FalseMinusTrueq')
plotfunc1(axs[2],HA_FalseMinusTrueq,'HA_FalseMinusTrueq')
plt.show()

p=stats.ttest_1samp(sum_FalseMinusTrueq,popmean=0).pvalue/2
print(f'sum_FalseMinusTrueq: p={p:.2f}')