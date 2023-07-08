"""
Analyse proprio data
New columns
    use_prop
    prop_outliers
    prop_stim: list of AU12 values for stimulus face (shown to participant)
    prop_resp: list of corresponding AU12 values for what participant did in response
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re, scipy.io
from acommonvars import *
import acommonfuncs

AU_to_plot = 'AU12'


def get_proprio_data(subject,AU_to_plot='AU12'):
    aulabels_list=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45'] 
    nAU = aulabels_list.index(AU_to_plot)
    contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\proprio*\\") #find the FF1 folder for this subject
    assert(len(contents)==1) 
    resultsFolder=contents[0]
    mat=scipy.io.loadmat(glob(f"{resultsFolder}*.mat")[0])
    data=mat['data'] #4D: blocks x trials x (stimface, ptface) x AU intensities for a single frame from OpenFace
    delays=mat['delays'] #array(nblocks,ntrials)
    data2 = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2],data.shape[3])) #collapse blocks and trials into one dimension
    stimface = data2[:,0,nAU]
    respface = data2[:,1,nAU]
    return stimface,respface


if __name__=='__main__':
    c = acommonfuncs.clock()
    t['use_prop'] = ((include) & (t.valid_proprioo==1)) 

    ### SETTABLE PARAMETERS 
    group = 'group02' #the grouping variable
    load_table=True

    ### Get data
    new_columns = ['use_prop','prop_stim', 'prop_resp']

    if load_table:
        t=acommonfuncs.add_table(t,'outcomes_prop.csv')
        t = acommonfuncs.str_columns_to_literals(t,['prop_stim', 'prop_resp'])
    else:
        t=acommonfuncs.add_columns(t,['prop_stim','prop_resp'])
        for i in range(len(t)):
            if t['use_prop'][i]:
                subject=t.subject[i]
                print(f'{c.time()[1]}: Subject {subject}')
                stimface,respface=get_proprio_data(subject,AU_to_plot) 
                t.at[i,'prop_stim'] = list(stimface)
                t.at[i,'prop_resp'] = list(respface)
        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_prop.csv')

#Calculate intercept, slope
# min, max, and range of AU12 displayed

subjects_to_exclude=['018'] #exclude these subjects, e.g. ['020']
"""
018 has zero values for AU12 all the time
"""

for i in range(len(t)):
    if t['use_prop'][i] and t.subject[i] not in subjects_to_exclude:
        subject=t.subject[i]
        print(f'{c.time()[1]}: Subject {subject}')
        t.at[i,'prop_stim_min'] = min(t.prop_stim[i])
        t.at[i,'prop_stim_max'] = max(t.prop_stim[i])
        t.at[i,'prop_stim_range'] = t.prop_stim_max[i] - t.prop_stim_min[i]
        coeffs=np.polyfit(t.prop_stim[i],t.prop_resp[i],1)
        t.at[i,'prop_slope'] = coeffs[0]
        t.at[i,'prop_intercept'] = coeffs[1]

def plot_subject(ax,i):
    predict_function = np.poly1d([t.prop_slope[i], t.prop_intercept[i]])
    ax.plot(t.prop_stim[i],t.prop_resp[i],'b.',t.prop_stim[i],predict_function(t.prop_stim[i]),'--k')
    ax.set_xlim([0,4])
    ax.set_ylim([0,4])
    ax.set_title(f"{t.subject[i]}, {t[group][i]}")

fig,axs=plt.subplots(6,8)
axs = axs.ravel()
j=0
for i in range(len(t)):
    if t['use_prop'][i] and t.subject[i] not in subjects_to_exclude:
        plot_subject(axs[j],i)
        j+=1
        if j==len(axs): break
fig.tight_layout()


plt.show()