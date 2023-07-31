"""
Analyse movieDI face data for Ricky (after OpenFace .csv files have been generated)
Mainly using the detailed.csv log file, and the OpenFace-processed .csv file with action unit time series
Resample action unit series from being indexed by frames as in the OpenFace .csv, to be indexed by time (sec) relative to the movie
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from acommonvars import *
import acommonfuncs
import amovie_utils


### SETTABLE PARAMETERS ###


target_fps=20 #target framerate to resample time series to
min_success = 0.95 #minimum proportion of successful webcam frames for a subject to be included. Default 0.95
movie_actual_start_time_sec = 2 #2sec for non MRI version, 10 sec for MRI version
movie_actual_duration_sec = 253 #253sec for Ricky stimulus, ??? for DISFA stimulus
gap_sec = 0 #use webcam outputs from this many seconds after movie starts, until this many seconds before movie ends. Default 0. Could set to 0.5 to avoid the first 0.5sec of the movie, and the last 0.5sec of the movie
start_time_sec = 2 + gap_sec #actual start time used for analysis
duration_sec = movie_actual_duration_sec - gap_sec
times_regular = np.arange(start_time_sec,duration_sec,1/target_fps)

c = acommonfuncs.clock()

if __name__=='__main__':

    group = 'group02' #the grouping variable
    load_array=True
    load_table=True
    outliers = []

    t['use_ricky'] = ((include) & (t.valid_movieo==1)) 
    t['ricky_outliers'] = t.subject.isin(outliers)


    valid_bool = (t.use_ricky) & (~t.ricky_outliers) #logical array of which subjects to include
    valid = np.where(valid_bool)[0] #indices in dataframe 't', of which subjects to include
    tgroup = t[group][valid_bool] #group ids within the subjects to include
    gps = list(tgroup.unique())
    if '' in gps: gps.remove('')


    if load_array:
        #t = acommonfuncs.add_table(t,'outcomes_ricky.csv')
        #t = acommonfuncs.str_columns_to_literals(t,['sinus_ts'])
        aussr = np.load(f'{temp_folder}\\ricky_aussr.npy') 
        ausdc = np.load(f'{temp_folder}\\ricky_ausdc.npy')
        #ausdr = np.load(f'{temp_folder}\\ricky_ausdr.npy')             
    else:
        get_all_resampled_time_series = lambda static_or_dynamic,r_or_c: amovie_utils.get_all_resampled_time_series(static_or_dynamic,r_or_c,t,valid,times_regular,aus_labels,c)    
        aussr = get_all_resampled_time_series('static','r')
        np.save(f'{temp_folder}\\ricky_aussr.npy',aussr)
        ausdc = get_all_resampled_time_series('dynamic','c')
        np.save(f'{temp_folder}\\ricky_ausdc.npy',ausdc)
        #ausdr = get_all_resampled_time_series('dynamic','r')
        #np.save(f'{temp_folder}\\ricky_ausdr.npy',ausdr)

    
    new_columns = ['use_ricky','ricky_outliers']
    for nAU in range(len(aus_labels)):
        action_unit = aus_labels[nAU]
        new_columns.append(f'ricky_sr_{action_unit}_mean')
        new_columns.append(f'ricky_sr_{action_unit}_cv')
        new_columns.append(f'ricky_dc_{action_unit}_mean')

    if load_table:
        t = acommonfuncs.add_table(t,'outcomes_ricky.csv')
    else:
        aussr_mean = np.mean(aussr,axis=1)
        aussr_cv = stats.variation(aussr,axis=1)
        ausdc_mean = np.mean(ausdc,axis=1)
        for i in range(len(valid)):
            t_index = valid[i]
            subject = t.subject[t_index]
            for nAU in range(len(aus_labels)):
                action_unit = aus_labels[nAU]
                t.at[t_index,f'ricky_sr_{action_unit}_mean'] = aussr_mean[i,nAU]
                t.at[t_index,f'ricky_sr_{action_unit}_cv'] = aussr_cv[i,nAU]
                t.at[t_index,f'ricky_dc_{action_unit}_mean'] = ausdc_mean[i,nAU]
        t.loc[:,new_columns].to_csv(f'{temp_folder}\\outcomes_ricky.csv')
    
    
    from sklearn import preprocessing
    #normalize on axis 1 to 0 mean and unit variance
    aussr_permute = np.transpose(aussr,[1,0,2])
    aussrn = preprocessing.StandardScaler().fit_transform(aussr_permute.reshape(aussr_permute.shape[0], -1)).reshape(aussr_permute.shape)
    aussrn = np.transpose(aussrn,[1,0,2])
    

    aussrn_mean = np.mean(aussrn,axis=0)
    #plot distance from mean time series, for each subject
    def distance(x,y):
        #this function, given two 2D arrays x and y, return a distance measure between them
        diff = x-y
        diff_abs_mean = np.mean(np.abs(diff))
        return diff_abs_mean
    dists = [distance(aussrn[i,:,:],aussrn_mean) for i in range(aussrn.shape[0])]
    z=pd.DataFrame(tgroup)
    z['dists'] = dists
    sns.stripplot(data=z,x='group02',hue='group02',palette=colors,y='dists') 
    plt.title('Distance from mean time series')

    
    nAU = np.where(np.array(aus_labels)=='AU12')[0][0]
    
    #K-means clustering
    array3D = aussrn
    array_stacked = np.reshape(array3D,[array3D.shape[0]*array3D.shape[1],array3D.shape[2]])
    #use sklearn to do kmeans clustering of 2D array array_stacked with 10 clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(init='k-means++',n_init='auto',n_clusters=10, random_state=0,algorithm='elkan').fit(array_stacked)
    #get the cluster labels for each row of array_stacked
    labels_1D = kmeans.labels_
    #reshape the labels array to be 3D
    labels = np.reshape(labels_1D,[array3D.shape[0],array3D.shape[1]])
    #get the mean of each cluster
    cluster_means = np.zeros([10,array3D.shape[2]])
    for cluster in range(10):
        cluster_means[cluster,:] = np.mean(array_stacked[labels_1D==cluster,:],axis=0)
    #get the most common value in each row of labels
    from scipy.stats import mode
    labels_mode = np.squeeze(mode(labels,axis=1).mode)
    

    '''
    #Plot sample time series
    
    plot_sample_time_series = lambda action_unit: amovie_utils.plot_sample_time_series(action_unit,aus_labels,times_regular,aussr,ausdr,ausdc,valid,t)
    plot_sample_time_series('AU12')
    plot_sample_time_series('AU17')
    
    
    #Plot group-mean time series with confidence interval
    
    aussr_12_long = amovie_utils.get_aus_df_long(aussr,'AU12',aus_labels,group,times_regular,tgroup)
    #ausdr_12_long = amovie_utils.get_aus_df_long(ausdr,'AU12',aus_labels,group,times_regular,tgroup)
    ausdc_12_long = amovie_utils.get_aus_df_long(ausdc,'AU12',aus_labels,group,times_regular,tgroup)
    fig,axs=plt.subplots(3)
    n_boot=10
    print(c.time()[1])
    sns.lineplot(data=aussr_12_long, x="time", y="AU12", ax=axs[0],hue=group,palette=colors, n_boot=n_boot)
    #sns.lineplot(data=ausdr_12_long, x="time", y="AU12", ax=axs[1],hue=group,palette=colors, n_boot=n_boot)
    sns.lineplot(data=ausdc_12_long, x="time", y="AU12", ax=axs[2],hue=group,palette=colors, n_boot=n_boot)
    axs[0].set_title('sr')
    axs[1].set_title('dr')
    axs[2].set_title('dc')
    print(c.time()[1])
    
    
    get_AU_df = lambda aus,outcome: amovie_utils.get_AU_df(aus,outcome,aus_labels,tgroup,group)
    df=get_AU_df(aussr,'AU12')
 
    
    #Scatterplots
    
    get_AU_df = lambda aus,outcome: amovie_utils.get_AU_df(aus,outcome,aus_labels,tgroup,group)
    acommonfuncs.pairplot(get_AU_df(aussr,'AU12'),vars=['AU12_mean','AU12_std'],height=1.5,robust=True,group=group,title='sr')
    #acommonfuncs.pairplot(get_AU_df(aussr,'AU12'),vars=['AU12_mean','AU12_std'],height=1.5,robust=True,group=group,title='dr')
    acommonfuncs.pairplot(get_AU_df(ausdc,'AU12'),vars=['AU12_mean','AU12_std'],height=1.5,robust=True,group=group,title='dc')
    #acommonfuncs.pairplot(get_AU_df(aussr,'AU12'),vars=['AU12_mean','AU12_median'],height=1.5,robust=True,group=group,title='sr')
    #acommonfuncs.pairplot(get_AU_df(aussr,'AU12'),vars=['AU12_std','AU12_mad'],height=1.5,robust=True,group=group,title='sr')
    

    #Compare across groups
    
    compare = lambda aus,outcome,title: amovie_utils.compare(aus,outcome,aus_labels,gps,tgroup,group,colors,title=title)
    compare(aussr,'mean','sr')
    compare(ausdc,'mean','dc')
    compare(aussr,'std','sr')
    
    compare(aussr,'cv','sr')
    #compare(aussr,'cv','dr')
    
    '''
    
    t2 = t.loc[t.use_ricky & (t[group]!='') & ~t.subject.isin(outliers),:]

    grid=acommonfuncs.pairplot(t2,x_vars=['ricky_sr_AU06_mean','ricky_sr_AU14_mean','ricky_dc_AU10_mean','ricky_dc_AU17_mean','ricky_sr_AU12_cv'],y_vars=['fsiq2'],height=1.5,kind='reg',robust=True,group=group)

    grid=acommonfuncs.pairplot(t2.loc[cc,:],x_vars=['ricky_sr_AU06_mean','ricky_sr_AU14_mean','ricky_dc_AU10_mean','ricky_dc_AU17_mean','ricky_sr_AU12_cv'],y_vars=['panss_bluntedaffect','panss_N','sofas','meds_chlor'],height=1.5,kind='reg',robust=True,group=group)
    


    plt.show(block=False)
    