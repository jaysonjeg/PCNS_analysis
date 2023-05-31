"""
Contains functions used by different scripts in roject_PCNS/Code_analysis
"""
import numpy as np, pandas as pd, datetime
from glob import glob
from acommonvars import *

class clock():
    """
    How to use
    c=acommon.clock()
    print(c.time())
    """
    def __init__(self):
        self.start_time=datetime.now()       
    def time(self):
        end_time=datetime.now()
        runtime=end_time-self.start_time
        runtime_sec = runtime.total_seconds()
        return runtime_sec,'{:.1f} sec.'.format(runtime_sec)

def get_beh_data(taskname,subject,suffix,use_MRI_task,header='infer'):
    """
    Get behavioural data from *out.csv in 'beh' folder
    suffix is 'out', 'detailed','PPG','face'. Which file to get within the subject's task folder
    Some tasks (cface1, movieDI) have two versions: one for MRI, one for non-MRI.
    if use_MRI_task==True, then get the MRI version, else get the non-MRI version of the task
    header should be set to None if first row of .csv is not column names
    """
    globstring = f"{data_folder}\\PCNS_{subject}_BL\\beh\\{taskname}*Ta_*\\"
    contents=glob(globstring) #'cface' task non-MRI folder for this particular subject
    if use_MRI_task:
        contents = [i for i in contents if 'Ta_M' in i]
    else:
        contents = [i for i in contents if 'Ta_M' not in i]
    if len(contents) != 1:
        print(f"ERROR: {len(contents)} folders found for {globstring}\n")
        assert(0)
    resultsFolder=contents[0]
    globstring = f"{resultsFolder}*{suffix}.csv"
    contents = glob(globstring)
    if len(contents)==1:
        df=pd.read_csv(contents[0],header=header) # make log csv into dataframe
        return df
    if len(contents) != 1:
        print(f"ERROR: {len(contents)} files found for {globstring}\n")
        return None

def get_openface_table(taskname,subject,static_or_dynamic,min_success=0.95):
    """Get the OpenFace intermediates .csv for this subject"""
    globstring = f"{intermediates_folder}\\openface_{taskname}\\{subject}"
    contents = glob(globstring)
    if len(contents) != 1:
        print(f"ERROR: {len(contents)} folders found for {globstring}")
        assert(0)
    resultsFolder=contents[0]
    face = pd.read_csv(glob(f"{resultsFolder}\\OpenFace_{static_or_dynamic}\\*_cam_20fps.csv")[0])
    all_frames=np.asarray(face['frame'])
    success=np.array(face[' success'])
    assert(np.sum(success)/len(success) > min_success) #check that most webcam frames are successful
    aus_labels_r = [f' {i}_r' for i in aus_labels] #Convert action unit labels into column labels for OpenFace .csv file
    aus_labels_c = [f' {i}_c' for i in aus_labels] 
    aus = face[aus_labels_r] #get all action units' time series for this subject. The rows are numbered from zero, whereas actual frames are numbered from 1. The error (1/20th of a second) is negligible.
    aus_c = face[aus_labels_c]
    aus.columns=aus_labels #rename columns, for example from ' AU01_r' to 'AU01'
    aus_c.columns=aus_labels #from ' AU01_c' to 'AU01'
    return all_frames,aus
'''
def get_redcap(redcap_file = "CogEmotPsych_DATA_2022-10-20_1804.csv"):  
    """
    return redcap table
    """

    redcap_folder = "C:\\Users\\c3343721\\Google Drive\\PhD\\Project_PCNS\\BackupRedcap"
    redcap_folder="C:\\Users\\Jayson\\Google Drive\\PhD\\Project_PCNS\\BackupRedcap"
    
    t=pd.read_csv(f"{redcap_folder}\\{redcap_file}")

    #data.query('pilotorreal==2 & group==2') #another way to get particular data

    healthy=((t.group==1) & (t.pilotorreal==2))
    clinical=((t.group==2) & (t.pilotorreal==2))

    healthy_attended=((healthy) & (t.attended==1))
    clinical_attended=((clinical) & (t.attended==1))

    healthy_didmri=((healthy) & (t.attended_fmri==1))
    clinical_didmri=((clinical) & (t.attended_fmri==1))
    
    return t
'''