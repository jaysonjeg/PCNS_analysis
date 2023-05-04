"""
Print out summary from redcap export .csv file

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

"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
import os, re


pc='laptop' #'laptop', 'home'
check_files=False #to check whether files match REDCAP 'valid' entries
check_numb_mri=False #to check number of files within each MRI folder (slow)
check_numb_beh=False
summary_printout=True 

if pc=='laptop':
    data_folder="Z:\\Shiami_DICOM\\Psychosis\\PCNS"
elif pc=='home':
    data_folder="Z:\\NEWYSNG\\Shiami_DICOM\\Psychosis\\PCNS"

redcap_folder="G:\\My Drive\\PhD\\Project_PCNS\\BackupRedcap"
redcap_file = "CogEmotPsych_DATA_2023-05-01_1814.csv"
IQ_cutoff = [80,150] #consider [80/85 to 120/125]

t=pd.read_csv(f"{redcap_folder}\\{redcap_file}")

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

### Check behavioural and MRI files ###

if check_files:
    print('Checking all files')
    if check_numb_mri: print('Checking no of files in mri too')
    if check_numb_beh: print('Checking no of files in beh too')
    def get(pattern,list_of_strings):
        return [s for s in list_of_strings if re.match(pattern,s)]
    def chk(folder,folder_items,sub,pattern,redcap_valid,check_no_of_files,nitems):
        """Big checker functon that makes sure saved beh and MRI files are correct"""
        temp = get(pattern,folder_items) #get all subfolder names that match pattern
        has_one_file = len(temp)==1 #whether there is exactly one matching subfolder
        check(redcap_valid,has_one_file,sub,f'{pattern}') #check whether has_one_file matches the corresponding entry in redcap e.g. 'valid_cfacei'
        if check_no_of_files and redcap_valid and has_one_file: #check whether the number of files within the subfolder matches 'nitems'
            if len(os.listdir(f"{folder}\\{temp[0]}"))!=nitems:
                print(f'{sub}: {pattern}: Wrong no. of files')
        return redcap_valid and has_one_file
    def check(a,b,sub,string):
        if a==True and b!=True: #2 goes to not True
            print(f'{sub}: {string}: Missing in folder but Redcap has it')
        elif b==True and a!=True:
            print(f'{sub}: {string}: Redcap says invalid but folder has it')

    z=[False]*len(t)
    df = pd.DataFrame({'cfaceo':z,'cfacei':z,'movieo':z,'moviei':z,'hrdo':z,'proprioo':z,'sinuso':z,'facegngi':z,'ffi':z,'mri':z,'diffusion':z},dtype=bool)

    data_folder_items=os.listdir(data_folder)
    invalid_but_has_folder=[]
    for i in range(len(t)):
        record_id=t.record_id[i]
        sub=f'{t.record_id[i]:03}'
        print(f'Checking sub {sub}')
        sub_folder=f"{data_folder}\\PCNS_{sub}_BL"
        has_sub_folder=os.path.isdir(sub_folder)
        if not(valid[i]) and has_sub_folder: invalid_but_has_folder.append(sub)        
        if valid[i]:
            print(f'Checking sub {sub}')
            check(t.attended[i]==1 , has_sub_folder , sub , 'has_sub_folder')
            if has_sub_folder or t.attended[i]==1:
                sub_folder_items=os.listdir(sub_folder)
                has_beh_folder = 'beh' in sub_folder_items
                check( t.attended[i]==1 , has_beh_folder , sub , 'has_beh_folder')
                if has_beh_folder or t.attended[i]==1:
                    beh_folder=f"{sub_folder}\\beh"
                    beh_folder_items=os.listdir(beh_folder)
                    chkb = lambda pattern,redcap_valid,nitems: chk(beh_folder,beh_folder_items,sub,pattern,redcap_valid,check_numb_beh,nitems)
                    
                    df.cfaceo[i]=chkb(r"cface1.*Ta_[HF]",t.valid_cfaceo[i],5)
                    df.cfacei[i]=chkb(r"cface1.*Ta_M",t.valid_cfacei[i],6)
                    df.facegngi[i]=chkb(r"facegng.*Ta_M",t.valid_facegngi[i],6)
                    df.ffi[i]=chkb(r"FF1.*Ta_M",t.valid_ffi[i],7)
                    df.hrdo[i]=chkb(r"^HRD",t.valid_hrdo[i],11)
                    df.moviei[i]=chkb(r"^movie.*Ta_M",t.valid_moviei[i],6)
                    df.movieo[i]=chkb(r"^movie.*Ta_[HF]",t.valid_movieo[i],4)
                    df.proprioo[i]=chkb(r"^proprio.*Ta",t.valid_proprioo[i],6)
                    df.sinuso[i]=chkb(r"^sinus.*Ta",t.valid_sinuso[i],2)
                    
                temp= [item.startswith('NEURO') for item in sub_folder_items]
                has_mri_folder = np.sum(temp)==1
                check(has_mri_folder , t.attended_fmri[i]==1 , sub , 'has_mri_folder')  
                if has_mri_folder or t.attended_fmri[i]==1:
                    mri_folder=f"{sub_folder}\\{sub_folder_items[temp.index(True)]}"
                    items=os.listdir(mri_folder)                    
                    chk2 = lambda pattern,redcap_valid,nitems: chk(mri_folder,items,sub,pattern,redcap_valid,check_numb_mri,nitems)                    
                    
                    df.facegngi[i] = df.facegngi[i] and\
                    chk2(r".*FACEGNG",t.valid_facegngi[i],542)     
                    
                    df.cfacei[i] = df.cfacei[i] and\
                    chk2(r".*BOLD_CFACE1_PA_0",t.valid_cfacei[i],665) and\
                    chk2(r".*BOLD_CFACE1_PA_PHYSIO",t.valid_cfacei[i],1) and \
                    chk2(r".*BOLD_CFACE1_PA_SBREF",t.valid_cfacei[i],1)                     
                    df.ffi[i] = df.ffi[i] and\
                    chk2(r".*BOLD_FF1_0",t.valid_ffi[i],545) and\
                    chk2(r".*BOLD_FF1_PHYSIO",t.valid_ffi[i],1) and\
                    chk2(r".*BOLD_FF1_SBREF",t.valid_ffi[i],1)
                    
                    df.moviei[i] = df.moviei[i] and\
                    chk2(r".*MOVIEDI_0",t.valid_moviei[i],327) and\
                    chk2(r".*MOVIEDI_PHYSIO",t.valid_moviei[i],1) and\
                    chk2(r".*MOVIEDI_SBREF",t.valid_moviei[i],1)
                    
                    df.mri[i] = \
                    chk2(r".*T1W_MPR_SAG_0",t.attended_fmri[i],208) and\
                    chk2(r".*FMAP_AP",t.attended_fmri[i],240) and\
                    chk2(r".*FMAP_PA",t.attended_fmri[i],240)
                    
                    df.diffusion[i] = \
                    chk2(r".*106DIR_0",t.valid_diffusion[i],107) and\
                    chk2(r".*106DIR_ADC",t.valid_diffusion[i],70) and\
                    chk2(r".*106DIR_COLFA",t.valid_diffusion[i],70) and\
                    chk2(r".*106DIR_FA",t.valid_diffusion[i],70) and\
                    chk2(r".*106DIR_TENSOR",t.valid_diffusion[i],1) and\
                    chk2(r".*106DIR_TRACEW",t.valid_diffusion[i],280) and\
                    chk2(r".*11B0",t.valid_diffusion[i],840)
                    
    print('Checked all files')
    print(f'Invalid pts with data folders are: {invalid_but_has_folder}')
else:
    df = pd.DataFrame({'cfaceo':t.valid_cfaceo==1,'cfacei':t.valid_cfacei==1,'movieo':t.valid_movieo==1,'moviei':t.valid_moviei==1,'hrdo':t.valid_hrdo==1,'proprioo':t.valid_proprioo==1,'sinuso':t.valid_sinuso==1,'facegngi':t.valid_facegngi==1,'ffi':t.valid_ffi==1,'mri':t.attended_fmri,'diffusion':t.attended_fmri},dtype=bool)

#### Quality control ###
print(f'Excluded {healthy_attended_exc.sum()} healthy and {clinical_attended_exc.sum()} clinical participants')


didmri = t.attended_fmri==1
didmri_and_future = ((didmri) & (future_subs))
assert(didmri_and_future.sum() == 0) #make sure everyone is either in the past or the future

assert(not(np.isnan(t.pilotorreal).any())) #make sure everyone has a value for pilotorreal
assert(not(np.isnan(t.group).any())) #make sure everyone has a value for group
assert(clinical_attended_all.sum()== (((t.dx_dsm___1)|(t.dx_dsm___2)|(t.dx_dsm___3)|(t.dx_dsm___4)|(t.dx_dsm___5)|(t.dx_dsm___6))).sum()) #make sure number of people with any diagnosis is equal to number of clinical pts who attended

temp =(attended) & ( np.isnan(didIQ) | np.isnan(t.valid_any) )
assert(temp.sum()==0) #make sure that all attended people have the required variables

print('ASSERTION COMMENTED OUT. PLS REVIEW')
#assert( attended.sum() == (~np.isnan(t.valid_any)).sum()) #that number of non-nan values in valid_any is equal to number of all attended people

assert( attended.sum() == didIQ.sum() ) #that number of ppl with didIQ is equal to number of all attended people



t2=pd.concat([t.record_id,attended,didIQ],axis=1) #View this. No rows should have attended==True and didIQ==False
t3=pd.concat([t.record_id,t.attended_fmri],axis=1)[attended] #View this. Only contains subjects with attended==1. 'attended_fmri' should have value 0 or 1, no NANs


def subgroup_summary(healthy_subgroup,clinical_subgroup):
    print(f"n: {healthy_subgroup.sum()}/{clinical_subgroup.sum()}")
    print(f"Mean age: {t.loc[healthy_subgroup,'age_years'].mean():.1f}/{t.loc[clinical_subgroup,'age_years'].mean():.1f}")
    print(f"healthy {(t.loc[healthy_subgroup,'sex']==1).sum()} M, {(t.loc[healthy_subgroup,'sex']==2).sum()} F ")
    print(f"clinical {(t.loc[clinical_subgroup,'sex']==1).sum()} M, {(t.loc[clinical_subgroup,'sex']==2).sum()} F ")


### Printout ###
if summary_printout:
    print(f"\nTotal Redcap entries: {t.shape[0]}")
    print(f"Total pilots: {(t.pilotorreal==1).sum()}")
    print(f"Total reals: {(t.pilotorreal==2).sum()}")

    print("\nWithin reals: Values are given as Healthy/Clinical")
    print(f"Redcap entries: {healthy.sum()}/{clinical.sum()}")

    print(f"\nWithin Redcap entries")
    print(f"future: {healthy_future.sum()}/{clinical_future.sum()}")
    print(f"past: did not attend: {healthy_nonattended.sum()}/{clinical_nonattended.sum()}")
    print(f"past: attended_exclude: {healthy_attended_exc.sum()}/{clinical_attended_exc.sum()}")
    print(f"past: attended_include: {healthy_attended_inc.sum()}/{clinical_attended_inc.sum()}")

    print(f"\npast did mri + future: {healthy_DidMRIPlusFuture.sum()}/{clinical_DidMRIPlusFuture.sum()}")
    print(f"past: did mri_include: {healthy_didmri_inc.sum()}/{clinical_didmri_inc.sum()}")
    tasks_of_interest = ['cfacei','ffi','facegngi','moviei','mri','diffusion']
    hdii=df.loc[healthy_didmri_inc,tasks_of_interest]
    cdii=df.loc[clinical_didmri_inc,tasks_of_interest]
    print(f"past: did mri_include_alltasks: {(hdii.sum(axis=1)==len(tasks_of_interest)).sum()}/{(cdii.sum(axis=1)==len(tasks_of_interest)).sum()}")

    print(f"\nFollowing values are within past subjects that were included and did mri, as well as future subjects")
    subgroup_summary(healthy_DidMRIPlusFuture,clinical_DidMRIPlusFuture)

    print(f"\nFollowing values are within past subjects that did not attend")
    subgroup_summary(healthy_nonattended,clinical_nonattended)

    print(f"\nFollowing values are within future subjects")
    subgroup_summary(healthy_future,clinical_future)

    print(f"\nFollowing values are within past subjects that attended, included, did mri")
    subgroup_summary(healthy_didmri_inc,clinical_didmri_inc)
    print(f"Mean IQ (plotted): {t.loc[healthy_didmri_inc,'fsiq2'].mean():.1f}/{t.loc[clinical_didmri_inc,'fsiq2'].mean():.1f}")

    print(f"\nOut of {clinical_didmri_inc.sum()} clinical pts:\
    \nSchizophrenia: {((clinical_didmri_inc) & (t.dx_dsm___1==1)).sum()}\
    \nSchizoaffective: {((clinical_didmri_inc) & (t.dx_dsm___2==1)).sum()}\
    \nBipolar: {((clinical_didmri_inc) & (t.dx_dsm___3==1)).sum()}\
    \nMDD: {((clinical_didmri_inc) & (t.dx_dsm___4==1)).sum()}\
    \nDelusional disorder: {((clinical_didmri_inc) & (t.dx_dsm___5==1)).sum()}\
    \nDrug-induced psychosis: {((clinical_didmri_inc) & (t.dx_dsm___6==1)).sum()}\
    ")

### PLOTS ###
fig,axs=plt.subplots(2,2)

ax=axs[0,0]
healthy_didmri_iq = t.loc[healthy_didmri_inc,'fsiq2']
clinical_didmri_iq = t.loc[clinical_didmri_inc,'fsiq2']
bins=np.linspace(65,135,int((135-65)/2.5))
ax.hist(healthy_didmri_iq,bins,alpha=0.5,label='HC_didmri')
ax.hist(clinical_didmri_iq,bins,alpha=0.5,label='PT_didmri')
ax.set_xlabel('IQ')
ax.legend()

ax=axs[1,0]
healthy_DidMRIPlusFuture_age = t.loc[healthy_DidMRIPlusFuture,'age_years']
clinical_DidMRIPlusFuture_age = t.loc[clinical_DidMRIPlusFuture,'age_years']
bins=np.linspace(15,55,int((55-15)/2.5))
ax.hist(healthy_DidMRIPlusFuture_age,bins,alpha=0.5,label='HC_didmri+future')
ax.hist(clinical_DidMRIPlusFuture_age,bins,alpha=0.5,label='PT_didmri+future')
ax.set_xlabel('Age (years)')
ax.legend()

ax=axs[0,1]
healthy_ppl, clinical_ppl = healthy_DidMRIPlusFuture, clinical_DidMRIPlusFuture
healthy_edu = [(t.loc[healthy_ppl,'edu_cat']==i).sum() for i in range(1,7)]
clinical_edu = [(t.loc[clinical_ppl,'edu_cat']==i).sum() for i in range(1,7)]
labels=['<Yr12','Yr12','NonUni','Bach','Mast','Doc']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
rects1 = ax.bar(x - width/2, healthy_edu, width, label='HC_DidMRIPlusFuture')
rects2 = ax.bar(x + width/2, clinical_edu, width, label='PT_DidMRIPlusFuture')
ax.set_xlabel('Education')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
fig.tight_layout()
"""
1, Didn't finish HS
2, High school
3, Non-university qualification
4, Bachelor's
5, Master's
6, Doctorate
"""

fig,axs=plt.subplots(2)
def plot0(array,n,title):
    axs[n].imshow(np.array(array))
    axs[n].set_aspect('auto')
    axs[n].set_title(title)
    axs[n].set_xlabel('Tasks')
    axs[n].set_ylabel('Subjects')
plot0(df[healthy_didmri_inc],0,'healthy')
plot0(df[clinical_didmri_inc],1,'clinical')
fig.suptitle('didmri_inc: missing data')
plt.show()


### Distribution of clinical factors in clinical participants ###

panss_labels_P = [f'panss_p{i}' for i in range(1,8)] #panss positive symptom scale
panss_labels_N = [f'panss_n{i}' for i in range(1,8)] #panss negative
panss_labels_G = [f'panss_g{i}' for i in range(1,17)] #panss general psychopathology
hamd_labels = [f'hamd_{i}' for i in range(1,18)] #hamilton depression scale
ymrs_labels = [f'ymrs_{i}' for i in range(1,12)] #young mania rating scale
sas_labels = [f'sas_{i}' for i in range(1,11)] #simpson angus scale for extreapyramidal symptoms
clinical_didmri_inc_panssP = t.loc[clinical_didmri_inc,panss_labels_P].sum(axis=1)
clinical_didmri_inc_panssN = t.loc[clinical_didmri_inc,panss_labels_N].sum(axis=1)
clinical_didmri_inc_panssG = t.loc[clinical_didmri_inc,panss_labels_G].sum(axis=1)
clinical_didmri_inc_hamd = t.loc[clinical_didmri_inc,hamd_labels].sum(axis=1)
clinical_didmri_inc_ymrs = t.loc[clinical_didmri_inc,ymrs_labels].sum(axis=1)
clinical_didmri_inc_sas = t.loc[clinical_didmri_inc,sas_labels].sum(axis=1)

clinical_didmri_chlor = t.loc[clinical_didmri_inc,'meds_chlor'] #chlorpromazine equivalents of antipsychotics
clinical_didmri_cgi = t.loc[clinical_didmri_inc,'cgi_s'] #clinician global impression of symptom severity
clinical_didmri_sofas = t.loc[clinical_didmri_inc,'sofas'] #sofas scale 'social and occupational functioning'

fig,axs=plt.subplots(3,3)
def plot1(data,n,xlabel,vertline=False):
    coords = np.unravel_index(n,(3,3))
    axs[coords].hist(data)
    axs[coords].set_xlabel(xlabel)
    axs[coords].set_aspect('auto')
plot1(clinical_didmri_inc_panssP,0,'PANSS Positive')
plot1(clinical_didmri_inc_panssN,1,'PANSS Negative')
plot1(clinical_didmri_inc_panssG,2,'PANSS General')
plot1(clinical_didmri_inc_hamd,3,'HAM-Depression')
plot1(clinical_didmri_inc_ymrs,4,'YMRS')
plot1(clinical_didmri_inc_sas,5,'Simpson Angus Scale')
plot1(clinical_didmri_chlor,6,'Chlorpromazine eq. (mg)')
plot1(clinical_didmri_cgi,7,'Clinical Global Impression')
plot1(clinical_didmri_sofas,8,'SOFAS')
fig.tight_layout()
plt.show()
