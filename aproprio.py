"""
Analyse facial proprioception task (proprio)
New columns
    use_prop
    prop_outliers
    prop_stim: list of AU12 values for stimulus face (shown to participant)
    prop_resp: list of corresponding AU12 values for what participant did in response
    prop_stim_min: minimum intensity for stimulus face
    prop_stim_max: maximum intensity
    prop_stim_range: prop_stim_max - prop_stim_min
    prop_slope: slope of linear regression of prop_resp vs prop_stim
    prop_intercept: intercept of regression
    prop_r2: goodness of fit of regression
"""
import numpy as np, pandas as pd, seaborn as sns, pingouin as pg
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


outliers=['018'] #exclude these subjects
"""
018 has zero values for AU12 all the time
"""


if __name__=='__main__':
    c = acommonfuncs.clock()

    ### SETTABLE PARAMETERS 
    group = 'group02' #the grouping variable
    load_table=False

    ### Get data
    new_columns = ['use_prop','prop_outliers','prop_stim', 'prop_resp','prop_stim_min','prop_stim_max','prop_stim_range','prop_slope','prop_intercept','prop_r2']

    if load_table:
        t = acommonfuncs.add_table(t,'outcomes_prop.csv')
        t = acommonfuncs.str_columns_to_literals(t,['prop_stim', 'prop_resp'])
    else:
        t['use_prop'] = ((include) & (t.valid_proprioo==1)) 
        t['prop_outliers'] = t.subject.isin(outliers)
        t=acommonfuncs.add_columns(t,['prop_stim','prop_resp'])
        for t_index in range(len(t)):
            if t['use_prop'][t_index]:
                subject=t.subject[t_index]
                print(f'{c.time()[1]}: Subject {subject}')
                stimface,respface=get_proprio_data(subject,AU_to_plot) 
                t.at[t_index,'prop_stim'] = list(stimface)
                t.at[t_index,'prop_resp'] = list(respface)

                t.at[t_index,'prop_stim_min'] = min(stimface)
                t.at[t_index,'prop_stim_max'] = max(stimface)
                t.at[t_index,'prop_stim_range'] = t.prop_stim_max[t_index] - t.prop_stim_min[t_index]

                if t.subject[t_index] not in outliers:
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stimface,respface)
                    t.at[t_index,'prop_slope'] = slope
                    t.at[t_index,'prop_intercept'] = intercept
                    t.at[t_index,'prop_r2'] = r_value**2


        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_prop.csv')

def plot_subject(ax,i):
    x = t.prop_stim[i]
    y = t.prop_resp[i]
    y_pred = np.poly1d([t.prop_slope[i],t.prop_intercept[i]])(x)
    ax.plot(x,y,'b.')
    ax.plot(x,y_pred,'--k')
    ax.set_xlim([0,4])
    ax.set_ylim([0,4])
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Response')
    ax.set_title(f"{t.subject[i]}, {t[group][i]}, slope {t.prop_slope[i]:.2f}, R2={t.prop_r2[i]:.2f}")

#Plot stimulus-response for individual subjects, and lines of best fit
fig,axs=plt.subplots(2,4)
axs = axs.ravel()
j=0
for t_index in range(len(t)):
    if t['use_prop'][t_index] and (t.subject[t_index] not in outliers) and (((t.prop_r2 < 0.5)[t_index]) | ((t.prop_slope < 0.65)[t_index])):
        plot_subject(axs[j],t_index)
        j+=1
        if j==len(axs): break
fig.tight_layout()

#Plot stimulus-response regression line for all subjects together
fig,ax=plt.subplots()
for t_index in range(len(t)):
    if t['use_prop'][t_index] and not(t['prop_outliers'][t_index]) and t[group][t_index]!='':
        x = t.prop_stim[t_index]
        y_pred = np.poly1d([t.prop_slope[t_index],t.prop_intercept[t_index]])(x)
        gp=t[group][t_index]
        ax.plot(x,y_pred,color=colors[gp],alpha=0.3,linewidth=0.5)
ax.set_xlim([0,4])
ax.set_ylim([0,4])
ax.set_xlabel('Stimulus')
ax.set_ylabel('Response')

#Compare outcome measures across groups

fig,axs=plt.subplots(2,3)
hue = group
t2 = t.loc[t.use_prop & (t[group]!='') & ~t.subject.isin(outliers),:]
sns.stripplot(ax=axs[0,0],data = t2, x=group, hue=hue,palette=colors,y='prop_stim_max')
sns.stripplot(ax=axs[1,0],data = t2, x=group, hue=hue,palette=colors,y='prop_slope')
sns.stripplot(ax=axs[1,1],data = t2, x=group, hue=hue,palette=colors,y='prop_intercept')
sns.stripplot(ax=axs[1,2],data = t2, x=group, hue=hue,palette=colors,y='prop_r2')
fig.tight_layout()

pg.ttest(t2.prop_slope[hc],t2.prop_slope[cc])['p-val']

#Scatter plots

grid=acommonfuncs.pairplot(t2,vars=['prop_stim_max','prop_slope','prop_intercept','prop_r2'],x_vars=None,y_vars=None,height=1.5,kind='reg',robust=True,group=group)
"""
grid=acommonfuncs.pairplot(t2,x_vars=['prop_stim_max','prop_slope','prop_intercept','prop_r2'],y_vars=['fsiq2'],height=1.5,kind='reg',robust=True,group=group)

grid=acommonfuncs.pairplot(t2.loc[cc,:],x_vars=['prop_stim_max','prop_slope','prop_intercept','prop_r2'],y_vars=['panss_P','panss_N','sofas','meds_chlor'],height=1.5,kind='reg',robust=True,group=group)
"""
plt.show(block=False)