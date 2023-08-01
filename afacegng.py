"""
Analyse facegng data
New columns are:
    use_facegng
    facegng_outliers
    gng_RT_fear: median reaction time in seconds for Fear Go trials
    gng_RT_calm

"""
import numpy as np, pandas as pd, seaborn as sns, pingouin as pg
import matplotlib.pyplot as plt
from glob import glob
import re

import acommonfuncs
from acommonvars import *

if __name__=='__main__':
    c = acommonfuncs.clock()
    t['use_facegng'] = ((include) & (t.valid_facegngi==1)) 


    ### SETTABLE PARAMETERS 
    group = 'group02' #the grouping variable
    load_table=False

    ### Get reaction time data
    new_columns = ['use_facegng', 'gng_RT_fear', 'gng_RT_calm']
    conds = ['fear target', 'calm target']
    if load_table:
        t=acommonfuncs.add_table(t,'outcomes_facegng.csv')
    else:
        for t_index in range(len(t)):
            if t['use_facegng'][t_index]:
                subject=t.subject[t_index]
                print(f'{c.time()[1]}: Subject {subject}')
                contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\facegng*\\")
                assert(len(contents)==1) 
                resultsFolder=contents[0]
                df=pd.read_csv(glob(f"{resultsFolder}*out.csv")[0]) # make log csv into dataframe
                for cond in conds:
                    inds = df.realcond == cond
                    t.at[t_index,f'gng_RT_{cond[:4]}'] = np.nanmedian(df.RT[inds])
        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_facegng.csv')

    t['gng_RT_FearMinusCalm'] = t['gng_RT_fear'] - t['gng_RT_calm']

    outliers_RT_fear_bool = pg.madmedianrule(t.loc[t.use_facegng & (t[group]!=''),'gng_RT_fear'])
    outliers_RT_fear = t.loc[t.use_facegng & (t[group]!=''),'subject'][outliers_RT_fear_bool].values
    outliers_RT_calm_bool = pg.madmedianrule(t.loc[t.use_facegng & (t[group]!=''),'gng_RT_calm'])
    outliers_RT_calm = t.loc[t.use_facegng & (t[group]!=''),'subject'][outliers_RT_calm_bool].values

    outliers = set(outliers_RT_fear).union(set(outliers_RT_calm))
    print(f'outliers_RT_fear: {outliers_RT_fear}')
    print(f'outliers_RT_calm: {outliers_RT_calm}')
    print(f'\nOutliers: {outliers}')

    t['facegng_outliers'] = t.subject.isin(outliers)

    fig,axs=plt.subplots(2,2)
    hue = group
    t2 = t.loc[t.use_facegng & (t[group]!='') & ~t.facegng_outliers,:]
    sns.stripplot(ax=axs[0,0],data = t2,x=group, hue=hue,y='gng_RT_fear')
    sns.stripplot(ax=axs[0,1],data = t2, x=group, hue=hue,y='gng_RT_calm')
    sns.stripplot(ax=axs[1,0],data = t2, x=group, hue=hue,y='gng_RT_FearMinusCalm')
    fig.tight_layout()
    plt.show(block=False)