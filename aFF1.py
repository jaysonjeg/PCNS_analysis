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
def qual(quality, data):
    #Return a subset of data where quality checks are passed
    newdata = [i for i,j in zip(data,quality) if j]
    return newdata
def plot_qual(axis,data,title):
    axis.hist(data)
    axis.set_title(title)
def plot_FalseTrue(axis,data,title):
    axis.hist(data)
    p=stats.ttest_1samp(data,popmean=0).pvalue/2
    axis.set_title(f'{title} \nmean={np.nanmean(data):.2f} p={p:.2f}')


def analyse_group_beh(subjects,groupname):
    print(f'Analysing group {groupname}')

    r = [get_data_table(i) for i in subjects] #r is a list of tuples of length 3. 
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
    fig.suptitle(f'{groupname}: Distribution of beh quality checks')
    plot_qual(axs[0],perc_trials_with_absent_ratings,'perc_trials_with_absent_ratings')
    plot_qual(axs[1],ANminusNE,'ANminusNE')
    plot_qual(axs[2],[int(i) for i in quality_all_possible_responses],'has_all_possible_responses')

    NE_FalseMinusTrueq = qual(quality, NE_FalseMinusTrue)
    AN_FalseMinusTrueq = qual(quality, AN_FalseMinusTrue)
    HA_FalseMinusTrueq = qual(quality, HA_FalseMinusTrue)
    ratings_NEq = qual(quality, ratings_NE)
    ratings_ANq = qual(quality, ratings_AN)
    ratings_HAq = qual(quality, ratings_HA)
    sum_FalseMinusTrueq = qual(quality, sum_FalseMinusTrue)

    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle(f'{groupname}: Distribution of False ratings minus True ratings')
    plot_FalseTrue(axs[0],NE_FalseMinusTrueq,'NE_FalseMinusTrueq')
    plot_FalseTrue(axs[1],AN_FalseMinusTrueq,'AN_FalseMinusTrueq')
    plot_FalseTrue(axs[2],HA_FalseMinusTrueq,'HA_FalseMinusTrueq')

    p=stats.ttest_1samp(sum_FalseMinusTrueq,popmean=0).pvalue/2
    print(f'{groupname}: sum_FalseMinusTrueq: p={p:.2f}')

    return NE_FalseMinusTrueq,AN_FalseMinusTrueq,HA_FalseMinusTrueq,ratings_NEq,ratings_ANq,ratings_HAq

task='FF1'

#files_with_task=glob(f"{data_folder}\\PCNS_*_BL\\beh\\{task}*\\")
#subjects_with_task=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task

HC=((healthy_didmri_inc) & (t.valid_ffi==1)) #healthy group
PT=((clinical_didmri_inc) & (t.valid_ffi==1)) #patient group
SZ = sz_didmri_inc #schizophrenia subgroup
SZA = sza_didmri_inc #schizoaffective subgroup
HC,PT,SZ,SZA = subs[HC],subs[PT],subs[SZ],subs[SZA]

HCdata = analyse_group_beh(HC[:],'HC')
#PTdata = analyse_group_beh(PT[:],'PT')
SZdata = analyse_group_beh(SZ[:],'SZ')

### Behavioural group contrasts ###
#t-test across groups, comparing (mean FalseMinusTrueDifference in each subject)
def print_t_FalseTrue(index,title):
    p=stats.ttest_ind(HCdata[index],SZdata[index],equal_var=True).pvalue
    mean = np.nanmean(HCdata[index])-np.nanmean(SZdata[index])
    print(f'SZ - HC: {title}: mean={mean:.2f} p={p:.2f}')
print_t_FalseTrue(0,'NE_FalseMinusTrueq')
print_t_FalseTrue(1,'AN_FalseMinusTrueq')
print_t_FalseTrue(2,'HA_FalseMinusTrueq')

def print_t_ratings_paired(index,title):
    #For each group, calculate mean rating for each face. Then do paired t-test to compare groups' mean ratings
    HCd = np.vstack(HCdata[index]) #becomes nsubjectsInGroup * nfacesInEmotionCategory
    SZd = np.vstack(SZdata[index])
    HCdm = np.nanmean(HCd,axis=0) #mean across subjectsInGroup
    SZdm = np.nanmean(SZd,axis=0)
    p_paired=stats.ttest_rel(HCdm,SZdm).pvalue
    #For each subject, calculate mean rating for that emotion. Then unpaired t-test across groups
    HCm = [np.nanmean(i) for i in HCdata[index]]
    SZm = [np.nanmean(i) for i in SZdata[index]]
    p_unpaired=stats.ttest_ind(HCm,SZm,equal_var=True).pvalue 
    mean = np.nanmean(SZdm-HCdm)
    print(f'SZ - HC: {title}: mean={mean:.2f}, paired p={p_paired:.2f}, unp p={p_unpaired:.2f}')
print_t_ratings_paired(3,'ratings_NEq, mean within group, then paired t-test to compare gps')
print_t_ratings_paired(4,'ratings_ANq, mean within group, then paired t-test to compare gps')
print_t_ratings_paired(5,'ratings_HAq, mean within group, then paired t-test to compare gps')

def plot_meanrating(axis,index,title):
    HCm = [np.nanmean(i) for i in HCdata[index]]
    SZm = [np.nanmean(i) for i in SZdata[index]]
    axis.hist(HCm,label='HC',alpha=0.5,color='blue')
    axis.hist(SZm,label='SZ',alpha=0.5,color='red')
    axis.set_title(title)
fig, axs = plt.subplots(nrows=1, ncols=3)
fig.suptitle(f'Distribution of ratings for each emotion')
plot_meanrating(axs[0],3,'ratings_NEq')
plot_meanrating(axs[1],4,'ratings_ANq')
plot_meanrating(axs[2],5,'ratings_HAq')

plt.show()