"""
Contains common functions and variables for Project_PCNS/Code_analysis

REDCAP variable names
pilotorreal: 1 pilot,2 real (covers all REDCAP entries)
group 1 control, 2 psychosis (covers all entries)
attended: blank or 1 
attended_fmri: blank or 0 or 1
valid_any: nan, 1 for valid, or 2 for invalid. Anyone with ((t.attended==1) & (t.pilotorreal==2)) has a non-nan entry for valid_any
age_years: number
sex: 1 male, 2 female
fsiq2: number

dx_notes_1   to   dx_notes_6:  diagnosis in notes
dx_dsm___1   to   dx_dsm__6: diagnosis after interview
1 schizophrenia, 2 schizoaffective, 3 bipolar, 4 MDD, 5 delusional disorder, 6 drug-induced psychosis

Education: 1, Didn't finish HS; 2, High school; 3, Non-university qualification; 4, Bachelor's; 5, Master's; 6, Doctorate
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

### SETTABLE PARAMETERS ###
pc='home' #'laptop', 'home'
files_source='NEWYSNG' #'local' for local machine, or else 'NEWYSNG'


if pc=='laptop':
    data_folder="Z:\\Shiami_DICOM\\Psychosis\\PCNS"
elif pc=='home':
    intermediates_folder='D:\\FORSTORAGE\\Data\\Project_PCNS\\intermediates'
    if files_source == 'NEWYSNG':
        data_folder="Z:\\NEWYSNG\\Shiami_DICOM\\Psychosis\\PCNS"
    elif files_source == 'local':
        data_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw"
redcap_folder="G:\\My Drive\\PhD\\Project_PCNS\\BackupRedcap"
redcap_file = "CogEmotPsych_DATA_2023-05-07_1336.csv"
IQ_cutoff = [80,150] #consider [80/85 to 120/125]

### SETUP ###

t=pd.read_csv(f"{redcap_folder}\\{redcap_file}")

subs=np.array([f'{t.record_id[i]:03}' for i in range(t.shape[0])])
attended=((t.pilotorreal==2) & t.attended==1)
dates=pd.to_datetime(t.attend_date,dayfirst=True)
future_subs = dates > datetime.today() #subjects coming in the future
valid = t.valid_any==1 
invalid = t.valid_any==2
didIQ=((attended) & (~np.isnan(t.fsiq2)))
include_IQ = ((attended) & (t.fsiq2 >= IQ_cutoff[0]) & (t.fsiq2 <= IQ_cutoff[1]))
exclude_IQ = ((attended) & ((t.fsiq2 < IQ_cutoff[0]) | (t.fsiq2 > IQ_cutoff[1]))) 
exclude_due_to_IQ = [i for i in np.where(exclude_IQ)[0] if i not in np.where(valid)[0]]
include = ((valid) & (include_IQ))
exclude = ((invalid) | (exclude_IQ))

healthy=((t.pilotorreal==2) & (t.group==1))
clinical=((t.pilotorreal==2) & (t.group==2))

healthy_attended_all=((healthy) & (t.attended==1))
clinical_attended_all=((clinical) & (t.attended==1))
healthy_didmri_all=((healthy) & (t.attended_fmri==1))
clinical_didmri_all=((clinical) & (t.attended_fmri==1))

healthy_attended_inc = ((healthy_attended_all) & (include)) #make sure these ppl have correct files
healthy_attended_exc = ((healthy_attended_all) & (exclude))
clinical_attended_inc = ((clinical_attended_all) & (include))
clinical_attended_exc = ((clinical_attended_all) & (exclude))

healthy_didmri_inc =  ((healthy_didmri_all) & (include))
clinical_didmri_inc =  ((clinical_didmri_all) & (include))

healthy_future=((healthy) & future_subs)
clinical_future = ((clinical) & future_subs)

healthy_nonattended=(healthy & (t.attended!=1) & ~(future_subs))
clinical_nonattended=(clinical & (t.attended!=1) & ~(future_subs))

healthy_DidMRIPlusFuture = (healthy_didmri_inc | healthy_future)
clinical_DidMRIPlusFuture = (clinical_didmri_inc | clinical_future)

sz_didmri_inc = (clinical_didmri_inc) & (t.dx_dsm___1==1) #schizophrenia, did mri
sza_didmri_inc= (clinical_didmri_inc) & (t.dx_dsm___2==1) #schizoaffective, did mri


panss_labels_P = [f'panss_p{i}' for i in range(1,8)] #panss positive symptom scamri_include_alltasksle
panss_labels_N = [f'panss_n{i}' for i in range(1,8)] #panss negative
panss_labels_G = [f'panss_g{i}' for i in range(1,17)] #panss general psychopathology
hamd_labels = [f'hamd_{i}' for i in range(1,18)] #hamilton depression scale
ymrs_labels = [f'ymrs_{i}' for i in range(1,12)] #young mania rating scale
sas_labels = [f'sas_{i}' for i in range(1,11)] #simpson angus scale for extreapyramidal symptoms
panss_P = t[panss_labels_P].sum(axis=1)
panss_N = t[panss_labels_N].sum(axis=1)
panss_G = t[panss_labels_G].sum(axis=1)
hamd = t[hamd_labels].sum(axis=1)
ymrs = t[ymrs_labels].sum(axis=1)
sas = t[sas_labels].sum(axis=1)
panss_bluntedaffect = t['panss_n1']
panss_anxiety = t['panss_g2']
panss_tension = t['panss_g4']
#'meds_chlor' for chlor equivalents, 'cgi_s' for CGI, 'sofas' for SOFAS

#### QUALITY CONTROL ###
print(f'Excluded {healthy_attended_exc.sum()} healthy and {clinical_attended_exc.sum()} clinical participants')
didmri = t.attended_fmri==1
didmri_and_future = ((didmri) & (future_subs))
assert(didmri_and_future.sum() == 0) #make sure everyone is either in the past or the future
assert(not(np.isnan(t.pilotorreal).any())) #make sure everyone has a value for pilotorreal
assert(not(np.isnan(t.group).any())) #make sure everyone has a value for group
assert(clinical_attended_all.sum()== (((t.dx_dsm___1)|(t.dx_dsm___2)|(t.dx_dsm___3)|(t.dx_dsm___4)|(t.dx_dsm___5)|(t.dx_dsm___6))).sum()) #make sure number of people with any diagnosis is equal to number of clinical pts who attended
temp =(attended) & ( np.isnan(didIQ) | np.isnan(t.valid_any) )
assert(temp.sum()==0) #make sure that all attended people have the required variables
assert( attended.sum() == (~np.isnan(t.valid_any)).sum()) #that number of non-nan values in valid_any is equal to number of all attended people
assert( attended.sum() == didIQ.sum() ) #that number of ppl with didIQ is equal to number of all attended people

t2=pd.concat([t.record_id,attended,didIQ],axis=1) #View this. No rows should have attended==True and didIQ==False
t3=pd.concat([t.record_id,t.attended_fmri],axis=1)[attended] #View this. Only contains subjects with attended==1. 'attended_fmri' should have value 0 or 1, no NANs



### FUNCTIONS ###

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