"""
Contains common variables for Project_PCNS/Code_analysis

REDCAP variable names
pilotorreal: 1 pilot,2 real (covers all REDCAP entries)
group 1 control, 2 psychosis (covers all entries)
attended: blank or 1 
attended_fmri: blank or 0 or 1
valid_any: nan, 1 for valid, or 2 for invalid. Anyone with ((t.attended==1) & (t.pilotorreal==2)) has a non-nan entry for valid_any
valid_cfacei, valid_moviei, valid_ffi, valid_facegngi, valid_cfaceo, valid_movieo, valid_hrdo, valid_sinuso, valid_proprioo, valid_eyetracking, valid_wasi, valid_diffusion
valid_cfacei means in-scanner, valid_cfaceo means out-of-scanner
age_years: number
sex: 1 male, 2 female
fsiq2: number

dx_notes_1   to   dx_notes_6:  diagnosis in notes
dx_dsm___1   to   dx_dsm__6: diagnosis after interview
1 schizophrenia, 2 schizoaffective, 3 bipolar, 4 MDD, 5 delusional disorder, 6 drug-induced psychosis

Education: 1, Didn't finish HS; 2, High school; 3, Non-university qualification; 4, Bachelor's; 5, Master's; 6, Doctorate
"""

import numpy as np, pandas as pd
from datetime import datetime

### SETTABLE PARAMETERS ###
"""
Sets 'data_folder', 'intermediates_folder', and 'redcap_file' to correct paths
"""


pc='home' #'laptop', 'home'
files_source='local' #'local' for local machine, or else 'NEWYSNG' for shared drive
if pc=='laptop' and files_source=='NEWYSNG':
    data_folder="Z:\\Shiami_DICOM\\Psychosis\\PCNS"
elif pc=='home':
    intermediates_folder='D:\\FORSTORAGE\\Data\\Project_PCNS\\intermediates'
    if files_source == 'NEWYSNG':
        data_folder="Z:\\NEWYSNG\\Shiami_DICOM\\Psychosis\\PCNS"
    elif files_source == 'local':
        data_folder="D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw"
redcap_file = "C:\\Users\\Jayson\OneDrive - The University Of Newcastle\\Drive\\PhD\\Project_PCNS\\BackupRedcap\\PCNS_redcap_data_table_01.csv"

'''
data_folder="G:\\My Drive\\Share_Angelica\\Data_raw\\per_subject"
intermediates_folder="G:\\My Drive\\Share_Angelica\\intermediates"
redcap_file = "G:\\My Drive\\Share_Angelica\\PCNS_redcap_data_table_01.csv"
'''
### CONSTANT VARIABLES ###
IQ_cutoff = [80,150] #exclude subjects outside this range. Default [80,150]. Consider [80/85 to 120/125]
aus_labels = ['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26'] #intentionally excluded AU45 which is blink
aus_names = ['InnerBrowRaiser','OuterBrowRaiser','BrowLowerer','UpperLidRaiser','CheekRaiser','LidTightener','NoseWrinkler','UpperLipRaiser','LipCornerPuller','Dimpler','LipCornerDepressor','ChinRaiser','LipStretcher','LipTightener','LipParts','JawDrop'] 
n_aus = len(aus_labels)

### SETUP ###
t=pd.read_csv(redcap_file)
subs=np.array([f'{t.record_id[i]:03}' for i in range(t.shape[0])]) #make record_ids from int to string, e.g. 3 to '003', 134 to '134'
t['subject'] = subs
attended=((t.pilotorreal==2) & t.attended==1) #not-pilot subject, and attended
dates=pd.to_datetime(t.attend_date,dayfirst=True)
future_subs = dates > datetime.today() #subjects coming in the future
valid = t.valid_any==1 #only use these subjects
invalid = t.valid_any==2 #do not use these subjects
didIQ=((attended) & (~np.isnan(t.fsiq2))) #subjects who did IQ test
include_IQ = ((attended) & (t.fsiq2 >= IQ_cutoff[0]) & (t.fsiq2 <= IQ_cutoff[1])) #those whose IQ was in the allowed range
exclude_IQ = ((attended) & ((t.fsiq2 < IQ_cutoff[0]) | (t.fsiq2 > IQ_cutoff[1]))) 
exclude_due_to_IQ = [i for i in np.where(exclude_IQ)[0] if i not in np.where(valid)[0]]
include = ((valid) & (include_IQ)) #only use these subjects (subset of 'valid')
exclude = ((invalid) | (exclude_IQ))

healthy=((t.pilotorreal==2) & (t.group==1))
clinical=((t.pilotorreal==2) & (t.group==2))
hc=healthy #alias
cc=clinical
sz = t.dx_dsm___1==1 #schizophrenia
sza = t.dx_dsm___2==1 #schizoaffective

t['group01']=''
for i in range(len(t)):
    if hc[i]: t.at[i,'group01']='hc'
    if sz[i]: t.at[i,'group01']='sz'
    if sza[i]: t.at[i,'group01']='sza'

hc_color='green'
cc_color='brown'
sza_color='orange'
sz_color='red'
colors={'hc':'green','cc':'brown','sza':'orange','sz':'red'}

healthy_attended_all=((healthy) & (t.attended==1)) #healthy subs who attended (valid and invalid)
clinical_attended_all=((clinical) & (t.attended==1))
healthy_didmri_all=((healthy) & (t.attended_fmri==1)) #healthy subs who did mri
clinical_didmri_all=((clinical) & (t.attended_fmri==1))

healthy_attended_inc = ((healthy_attended_all) & (include)) #healthy subs who attended and are 'valid'
healthy_attended_exc = ((healthy_attended_all) & (exclude))
clinical_attended_inc = ((clinical_attended_all) & (include))
clinical_attended_exc = ((clinical_attended_all) & (exclude))
healthy_didmri_inc =  ((healthy_didmri_all) & (include))
clinical_didmri_inc =  ((clinical_didmri_all) & (include))
sz_didmri_inc = (clinical_didmri_inc) & (t.dx_dsm___1==1) #schizophrenia, did mri
sz_attended_inc = (clinical_attended_inc) & (t.dx_dsm___1==1) #schizophrenia, attended
sza_didmri_inc= (clinical_didmri_inc) & (t.dx_dsm___2==1) #schizoaffective, did mri
sza_attended_inc= (clinical_attended_inc) & (t.dx_dsm___2==1) #schizoaffective, attended

panss_labels_P = [f'panss_p{i}' for i in range(1,8)] #panss positive symptom 
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