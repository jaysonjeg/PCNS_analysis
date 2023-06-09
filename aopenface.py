"""
Code using OpenFace on facial videos from PCNS study (movieDI and cface) to produce .csv files of action unit time series. Checks the intermediates folder and only does the subjects who haven't been extracted yet. If you Ctrl+Z to kill this script while it's running, make sure you delete the last-created folder (which would have a 0-byte .csv). Then next time you can just run it and it will continue from where it left off
"""

import matplotlib.pyplot as plt, numpy as np, pandas as pd
from glob import glob
import re, os, subprocess
from acommonvars import *

#Settable parameters
taskname='cface1_*_Ta_*' #'movieDI_*_Ta_F_Ricky*'   ,   'cface1_*_Ta_H*'

run_duration = np.inf #stop after this time (seconds)
show_plot=False #show AU 12 time series per participant

task_dict={'movieDI_*_Ta_*_Ricky*':'movieDI', 'cface1_*_Ta_*':'cface1'} #mapping from 'taskname' to the label in OpenFace output file


if pc=='laptop':
    openfacefolder='C:/Users/c3343721/Desktop/FaceThings/OpenFace-master' 
elif pc=='home':
    openfacefolder='D:/FORSTORAGE/OpenFace-master'
openfacefile=f'{openfacefolder}/OpenFace-master/x64/Release/FeatureExtraction.exe'

"""
print(len(files_with_task))
print(files_with_task[0:2])
assert(0)
"""

files_with_task=glob(f"{data_folder}\\PCNS_*_BL\\beh\\{taskname}")
files_with_task_and_video=glob(f"{data_folder}\\PCNS_*_BL\\beh\\{taskname}\\*.avi")
files_with_task = [i for i in files_with_task if 'Ta_M' not in i]
files_with_task_and_video = [i for i in files_with_task_and_video if 'Ta_M' not in i]

assert(len(files_with_task)==len(files_with_task_and_video))
subjects=[re.search('PCNS_(.*)_BL',file).groups()[0] for file in files_with_task] #gets all subject names who have data for the given task

subjects_to_exclude=['003','004'] #exclude these subjects
"""
004 (is this pilot subject?) is 30fps, everyone else is 20fps. Seems to go beyond 100%...
"""
subjects_with_task = [subject for subject in subjects if subject not in subjects_to_exclude]
import acommonfuncs
c=acommonfuncs.clock()

### RUN OPENFACE TO GET .CSV FILES and put them in folder 'intermediates' ###
for subject in subjects_with_task:
    for extract_type in ['static','dynamic']:

        out_folder=f'{intermediates_folder}\\openface_{task_dict[taskname]}\\{subject}\\OpenFace_{extract_type}'
        if os.path.exists(out_folder):
            print(f'{c.time()[1]}: {subject} {extract_type} ALREADY EXISTS')
        else:
            print(f'{c.time()[1]}: {subject} {extract_type} STARTING...')            
            if c.time()[0] > run_duration: 
                print('Ran out of time')
                break
            
            contents=glob(f"{data_folder}\\PCNS_{subject}_BL\\beh\\{taskname}\\*.avi")
            if len(contents)!=1:
                print(f'We didnt find exactly 1 video in: {subject} {taskname}')
                assert(0)            
            vid_path=contents[0] 
            if extract_type=='static':
                command=f'{openfacefile} -f {vid_path} -au_static -out_dir {out_folder} -aus'
            elif extract_type=='dynamic':
                command=f'{openfacefile} -f {vid_path} -out_dir {out_folder} -aus'
                
            subprocess.call(command)


### Plot each subject's AU12 trace on different subplot ###
if show_plot:
    fig,ax=plt.subplots() 
    for subject in subjects_with_task[0:10]:
        out_folder=f'D:\\FORSTORAGE\\Data\\Project_PCNS\\intermediates\\{subject}\\{task_dict[taskname]}\\OpenFace'
        face_path=glob(f"{out_folder}\\*.csv")[0]
        face=pd.read_csv(face_path)
        ax.plot(np.asarray(face[' timestamp']),np.asarray(face[' AU12_r']),label=subject)
        
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (s)')
    ax.legend()
    plt.show()
