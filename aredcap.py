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

Education: 1, Didn't finish HS; 2, High school; 3, Non-university qualification; 4, Bachelor's; 5, Master's; 6, Doctorate
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
import os, re
import aredcap_utils


### SETTABLE PARAMETERS ###

pc='home' #'laptop', 'home'
check_files=True #to check whether files match REDCAP 'valid' entries
check_numb_mri=False #to check number of files within each MRI folder (slow)
check_numb_beh=True
summary_printout=False 
plot_demographics=False

if pc=='laptop':
    data_folder="Z:\\Shiami_DICOM\\Psychosis\\PCNS"
elif pc=='home':
    data_folder="Z:\\NEWYSNG\\Shiami_DICOM\\Psychosis\\PCNS"

redcap_folder="G:\\My Drive\\PhD\\Project_PCNS\\BackupRedcap"
redcap_file = "CogEmotPsych_DATA_2023-05-07_1336.csv"
IQ_cutoff = [80,150] #consider [80/85 to 120/125]


### SETUP ###

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

### Check behavioural and MRI files ###

#Anomalies where number of data files is unexpected
data_anomalies = {'012':'facegng no eye',\
                '013':'movieRicky ok but has heart, and empty summary.csv',\
            '031': 'cface nonMRI: no heart',\
            '032': 'facegng no eye',\
            '058': 'facegng no eye',\
            '065': 'facegng and FF1 no eye',\
            '108': 'extra fmaps for task cface',\
            '130': 'FF1 no eye',\
                }

"""
for sub in ['005','011','040']:
    data_anomalies[sub]='no eye' #can remove these once I get an updated REDCAP
"""
if check_files:
    df=aredcap_utils.do_check_files(t,check_numb_mri,check_numb_beh,data_folder,valid,data_anomalies)
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

assert( attended.sum() == (~np.isnan(t.valid_any)).sum()) #that number of non-nan values in valid_any is equal to number of all attended people
assert( attended.sum() == didIQ.sum() ) #that number of ppl with didIQ is equal to number of all attended people

t2=pd.concat([t.record_id,attended,didIQ],axis=1) #View this. No rows should have attended==True and didIQ==False
t3=pd.concat([t.record_id,t.attended_fmri],axis=1)[attended] #View this. Only contains subjects with attended==1. 'attended_fmri' should have value 0 or 1, no NANs



### Demographics Printout ###

if summary_printout:

    def subgroup_summary(healthy_subgroup,clinical_subgroup,iq=False,do_stats=False):
        print(f"n: {healthy_subgroup.sum()}/{clinical_subgroup.sum()}")
        print(f"healthy {(t.loc[healthy_subgroup,'sex']==1).sum()} M, {(t.loc[healthy_subgroup,'sex']==2).sum()} F ")
        print(f"clinical {(t.loc[clinical_subgroup,'sex']==1).sum()} M, {(t.loc[clinical_subgroup,'sex']==2).sum()} F ")
        print(f"Mean age: {t.loc[healthy_subgroup,'age_years'].mean():.1f}/{t.loc[clinical_subgroup,'age_years'].mean():.1f}")
        if iq:
            print(f"Mean IQ (plotted): {t.loc[healthy_subgroup,'fsiq2'].mean():.1f}/{t.loc[clinical_subgroup,'fsiq2'].mean():.1f}")

        if do_stats:
            def do_t_test(x,y,string):
                from scipy.stats import ttest_ind
                t,p = ttest_ind(x,y,equal_var=False)
                print(f'{string}, t-test: p={p:.2f}')
            def do_chi_squared_test(x,y,string):
                from scipy.stats import chi2_contingency
                possible_values = np.unique(x)
                contingency_table = [[ (x==value).sum() for value in possible_values], [ (y==value).sum() for value in possible_values]]
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                print(f'{string}, chi2: p={p:.2f}')
            do_t_test(t.loc[healthy_subgroup,'fsiq2'],t.loc[clinical_subgroup,'fsiq2'],'IQ')
            do_t_test(t.loc[healthy_subgroup,'age_years'],t.loc[clinical_subgroup,'age_years'],'Age')
            do_chi_squared_test(t.loc[healthy_subgroup,'sex'],t.loc[clinical_subgroup,'sex'],'Sex')
            do_chi_squared_test(t.loc[healthy_subgroup,'edu_cat'],t.loc[clinical_subgroup,'edu_cat'],'Education')


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
    subgroup_summary(healthy_didmri_inc,clinical_didmri_inc,iq=True,do_stats=True)

    sz_didmri_inc = (clinical_didmri_inc) & (t.dx_dsm___1==1)
    sza_didmri_inc= (clinical_didmri_inc) & (t.dx_dsm___2==1)

    print(f"\nOut of {clinical_didmri_inc.sum()} clinical pts:\
    \nSchizophrenia: {sz_didmri_inc.sum()}\
    \nSchizoaffective: {sza_didmri_inc.sum()}\
    \nBipolar: {((clinical_didmri_inc) & (t.dx_dsm___3==1)).sum()}\
    \nMDD: {((clinical_didmri_inc) & (t.dx_dsm___4==1)).sum()}\
    \nDelusional disorder: {((clinical_didmri_inc) & (t.dx_dsm___5==1)).sum()}\
    \nDrug-induced psychosis: {((clinical_didmri_inc) & (t.dx_dsm___6==1)).sum()}\
    ")

if plot_demographics:

    ### Demographics plotted for both groups ###

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

    ### Distribution of clinical factors in clinical participants ###         
    
    aredcap_utils.plot_clinical_factors_for_group(t,clinical_didmri_inc,'clinical_didmri_inc',panss_P,panss_N,panss_G,hamd,ymrs,sas )
    aredcap_utils.plot_clinical_factors_for_group(t,sz_didmri_inc,'sz_didmri_inc',panss_P,panss_N,panss_G,hamd,ymrs,sas )
    aredcap_utils.plot_clinical_factors_for_group(t,sza_didmri_inc,'sza_didmri_inc',panss_P,panss_N,panss_G,hamd,ymrs,sas )
    plt.show()    
