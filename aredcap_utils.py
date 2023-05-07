import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, re

    
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
        print(f'{sub}: {string}: Redcap has it but wrong no of folders')
    elif b==True and a!=True:
        print(f'{sub}: {string}: Redcap says invalid but folder has it')

def do_check_files(t,check_numb_mri,check_numb_beh,data_folder,valid,anomalies):

    print('Checking all files')
    if check_numb_mri: print('Checking no of files in mri too')
    if check_numb_beh: print('Checking no of files in beh too')

    z=[False]*len(t)
    df = pd.DataFrame({'cfaceo':z,'cfacei':z,'movieo':z,'moviei':z,'hrdo':z,'proprioo':z,'sinuso':z,'facegngi':z,'ffi':z,'mri':z,'diffusion':z},dtype=bool)

    invalid_but_has_folder=[]
    for i in range(len(t)):
        sub=subs[i]
        sub_folder=f"{data_folder}\\PCNS_{sub}_BL"
        has_sub_folder=os.path.isdir(sub_folder)

        if not(valid[i]) and has_sub_folder: 
            print(f'sub {sub} invalid but has folder')
            invalid_but_has_folder.append(sub)        
        if valid[i]:
            print(f'sub {sub}: checking')
            if sub in anomalies.keys():
                print(f'{sub}: {anomalies[sub]}')
            check(t.attended[i]==1 , has_sub_folder , sub , 'has_sub_folder')
            if has_sub_folder or t.attended[i]==1:
                sub_folder_items=os.listdir(sub_folder)
                has_beh_folder = 'beh' in sub_folder_items
                check( t.attended[i]==1 , has_beh_folder , sub , 'has_beh_folder')
                if has_beh_folder or t.attended[i]==1:
                    beh_folder=f"{sub_folder}\\beh"
                    beh_folder_items=os.listdir(beh_folder)
                    chkb = lambda pattern,redcap_valid,nitems: chk(beh_folder,beh_folder_items,sub,pattern,redcap_valid,check_numb_beh,nitems)
                    if t.loc[i,'valid_eyetracking']==1:
                        nfiles_cfacei,nfiles_facegngi,nfiles_ffi=6,6,7
                    else:
                        nfiles_cfacei,nfiles_facegngi,nfiles_ffi=5,5,6
                    df.cfaceo[i]=chkb(r"cface1.*Ta_[HF]",t.valid_cfaceo[i],5)
                    df.cfacei[i]=chkb(r"cface1.*Ta_M",t.valid_cfacei[i],nfiles_cfacei)

                    df.facegngi[i]=chkb(r"facegng.*Ta_M",t.valid_facegngi[i],nfiles_facegngi)
                    df.ffi[i]=chkb(r"FF1.*Ta_M",t.valid_ffi[i],nfiles_ffi)
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
    return df





def plot_clinical_factors_for_group(t,subs,title_prefix,panss_P,panss_N,panss_G,hamd,ymrs,sas):
    subs_panssP = panss_P[subs]
    subs_panssN = panss_N[subs]
    subs_panssG = panss_G[subs]
    subs_hamd = hamd[subs]
    subs_ymrs = ymrs[subs]
    subs_sas = sas[subs]

    subs_chlor = t.loc[subs,'meds_chlor'] #chlorpromazine equivalents of antipsychotics
    subs_cgi = t.loc[subs,'cgi_s'] #clinician global impression of symptom severity
    subs_sofas = t.loc[subs,'sofas'] #sofas scale 'social and occupational functioning'

    fig,axs=plt.subplots(3,3)
    def plot1(data,n,xlabel,vertline=False):
        coords = np.unravel_index(n,(3,3))
        axs[coords].hist(data)
        axs[coords].set_xlabel(xlabel)
        axs[coords].set_title(f'{np.mean(data):.1f}')
        axs[coords].set_aspect('auto')
    plot1(subs_panssP,0,'PANSS Positive')
    plot1(subs_panssN,1,'PANSS Negative')
    plot1(subs_panssG,2,'PANSS General')
    plot1(subs_hamd,3,'HAM-Depression')
    plot1(subs_ymrs,4,'YMRS')
    plot1(subs_sas,5,'Simpson Angus Scale')
    plot1(subs_chlor,6,'Chlorpromazine eq. (mg)')
    plot1(subs_cgi,7,'Clinical Global Impression')
    plot1(subs_sofas,8,'SOFAS')
    fig.suptitle(f'{title_prefix}: clinical factors')
    fig.tight_layout()