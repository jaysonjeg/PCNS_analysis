"""
Analyse movieDI data (after aopenface.py)
"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re


taskname='movieDI_*_Ta_F_Ricky*'
task_dict={'movieDI_*_Ta_F_Ricky*':'movieDI'} #mapping from 'taskname' to the label in OpenFace output file
top_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw\\"

#Search relevant subject names in Data_raw/..beh..
files_with_task=glob(f"{top_folder}\\PCNS_*_BL\\beh\\{taskname}\\")
files_with_task_and_video=glob(f"{top_folder}\\PCNS_*_BL\\beh\\{taskname}\\*.avi")
assert(len(files_with_task)==len(files_with_task_and_video))
subjects=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task
subjects_to_exclude=['004','005','008'] #exclude these subjects
"""
004 (is this pilot subject?) is 30fps, everyone else is 20fps
"""
subjects_with_task = [subject for subject in subjects if subject not in subjects_to_exclude] 

#Search relevant subject names in 'intermediates/../movieDI/OpenFace
files_with_FaceCSV=glob(f'D:\\FORSTORAGE\\Data\\Project_PCNS\\intermediates\\*\\{task_dict[taskname]}\\OpenFace\\*.csv')
subjects=[re.search(f'intermediates\\\\(.*)\\\\{task_dict[taskname]}\\\\OpenFace',file).groups()[0] for file in files_with_FaceCSV]
subjects_with_FaceCSV = [subject for subject in subjects if subject not in subjects_to_exclude] 

subjects_with_FaceCSV=subjects_with_FaceCSV[0:5]

#should check that subjects_with_FaceCSV is equal or subset of subjects_with_task

oldtime_allsubs=[]
AU12_allsubs=[]
time_interp=[]
AU12_interp_allsubs=[]

for subject in subjects_with_FaceCSV:
    print(subject)

    data_path=glob(f"{top_folder}\\PCNS_{subject}_BL\\beh\\{taskname}\\*detailed.csv")[0]
    face_path=glob(f'D:\\FORSTORAGE\\Data\\Project_PCNS\\intermediates\\{subject}\\{task_dict[taskname]}\\OpenFace\\*.csv')[0]

    """
    sub_path='D:\\FORSTORAGE\\Data\Project_PCNS\\Data\JJ'
    file_prefix='movieDI_JJ_Te_F_DISFAVideoStimulus_to20s_2022_Feb_07_1418'

    data_path='{}\\{}_detailed.csv'.format(sub_path,file_prefix)
    face_path='{}\\OpenFace\\{}_cam_30fps.csv'.format(sub_path,file_prefix)
    """
    data=pd.read_csv(data_path) #contains ptframenum, fliptime
    face=pd.read_csv(face_path) #contains OpenFace outputs

    #Get estimate of actual time for each row of OpenFace output csv
    face['oldtime']=np.nan
    for index,row in face.iterrows(): #for each row of OpenFace output (corresponding to webcam frames)
        framenum=row['frame'] 
        try: #if that webcam frame coincides with any movie frames (appears in data.ptframenums)
            fliptime=data.fliptimes[data.ptframenums==framenum].iloc[0] #find the first movie fliptime which corresponds to that webcam frame
            #Put this fliptime into the 'face' dataframe
            face.at[index,'oldtime']=fliptime
        except:
            pass

    nonNANinds=~np.isnan(face.oldtime)
    successes=face[' success'][nonNANinds] 
    successRate = 100 * successes.sum() / len(successes) #   percent of frames where OpenFace detected a face (within frames that were during the movie, as opposed to before or after)

    oldtime=np.asarray(face['oldtime'][nonNANinds])
    AU12=np.asarray(face[' AU12_r'][nonNANinds])
    
    oldtime_allsubs.append(oldtime)
    AU12_allsubs.append(AU12)
    
    #Interpolation to new timestamps separated by 0.05s
    from scipy.interpolate import interp1d
    time_interp=np.arange(np.ceil(min(data.fliptimes)),np.floor(max(data.fliptimes)),0.2)
    func=interp1d(oldtime,AU12)
    AU12_interp=func(time_interp)
    AU12_interp_allsubs.append(AU12_interp)
    
    
    """
    #Plot AU12 time series, each participant at a time
    fig,ax=plt.subplots()
    ax.plot(oldtime,AU12,label='raw')
    ax.plot(time_interp,AU12_interp,label='interpolated')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (s)')
    ax.set_title(f"Sub {subject} success {successRate:.1f}%")
    ax.legend()
    plt.show()
    """


#Plot AU12 time series, everyone together
fig,ax=plt.subplots()
for i in range(len(subjects_with_FaceCSV)):
    subject=subjects_with_FaceCSV[i]
    #ax.plot(oldtime_allsubs[i],AU12_allsubs[i],label=f'raw sub {subject}')
    ax.plot(time_interp,AU12_interp_allsubs[i],label=f'interp sub {subject}')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_xlabel('Time (s)')
ax.set_ylabel('AU12')
ax.set_title(f"Watching Ricky")
ax.legend()
plt.show()
