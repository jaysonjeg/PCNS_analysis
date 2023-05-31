"""
For every subfolder 'sub_folder' in 'source_folder', check if that subfolder exists in 'target_folder'. If it does exist in target_folder, print f"{sub_folder} exists". If it doesn't exist, print f"{sub_folder doesnt exist}  Copying now"} and then copy sub_folder and all of its contents to target_folder.
"""

import os, re, shutil

'''
#Copy files from NEWYSNG to local machine
source_folder="Z:\\NEWYSNG\\Shiami_DICOM\\Psychosis\\PCNS\\Data\\Data_analysis"
target_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw"
for sub_folder in os.listdir(source_folder):
    if re.match(r'PCNS_(.*)_BL',sub_folder):
        if os.path.exists(f'{target_folder}\\{sub_folder}'):
            pass
            #print(f'{sub_folder} exists')
        else:
            print(f'{sub_folder} doesnt exist. Copying now')
            #shutil.copytree(f'{source_folder}\\{sub_folder}',f'{target_folder}\\{sub_folder}')
            #print(f'{sub_folder[5:8]}',end=', ')
'''
'''
#Copy USB files from local machine to NEWYSNG
source_folder = "D:\\FORSTORAGE\\Data\\Project_PCNS\\intermediates\\openface_movieDI"
target_folder="Z:\\NEWYSNG\\PCNS\\Data\\Data_analysis\\openface_movieDI"
for sub_folder in os.listdir(source_folder):
    if os.path.exists(f'{target_folder}\\{sub_folder}'):
        print(f'{sub_folder} exists')
    else:
        print(f'{sub_folder} doesnt exist. Copying now')
        shutil.copytree(f'{source_folder}\\{sub_folder}',f'{target_folder}\\{sub_folder}')
        #print(f'{sub_folder[5:8]}',end=', ')
'''

'''
#Copy OpenFace output files from per_subject folder to task-dedicated folder
source_folder = 'Z:\\NEWYSNG\\PCNS\\Data\\Data_analysis\\OpenFace_outputs'
target_folder = 'Z:\\NEWYSNG\\PCNS\\Data\\Data_analysis\\openface_movieDI'
for sub_folder in os.listdir(source_folder):
    if os.path.exists(f'{source_folder}\\{sub_folder}\\movieDI'):
        print(f'{sub_folder} has movieDI')
        if os.path.exists(f'{target_folder}\\{sub_folder}'):
            print(f'{sub_folder} exists')
        else:
            print(f'{sub_folder} doesnt exist. Copying now')
            shutil.copytree(f'{source_folder}\\{sub_folder}\\movieDI',f'{target_folder}\\{sub_folder}')
    else:
        print(f'{sub_folder} doesnt have movieDI')
'''

#Copy out files for a single task, leaving the rest
from glob import glob
taskname='movieDI'
use_MRI_task=False #MRI or non MRi version of the task
source_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw"
target_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw_only_movie_Ricky"
for sub_folder in os.listdir(source_folder):
    if re.match(r'PCNS_(.*)_BL',sub_folder):
        if os.path.exists(f'{target_folder}\\{sub_folder}'):
            print(f'{sub_folder} exists')
        else:
            print(f'{sub_folder} doesnt exist. ')
            globstring = f"{source_folder}\\{sub_folder}\\beh\\{taskname}*Ta_*"
            contents=glob(globstring) 
            if use_MRI_task:
                contents = [i for i in contents if 'Ta_M' in i]
            else:
                contents = [i for i in contents if 'Ta_M' not in i] 
            if len(contents) == 1:
                resultsFolder=contents[0]
                print(f'{sub_folder}: Copying')
                start_index = re.search(taskname,resultsFolder).span(0)[0]
                foldername = resultsFolder[start_index:]
                shutil.copytree(resultsFolder,f'{target_folder}\\{sub_folder}\\beh\\{foldername}',ignore=shutil.ignore_patterns('*.avi')) #don't copy files ending in .avi