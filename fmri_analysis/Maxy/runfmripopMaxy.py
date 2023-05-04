"""
Run this in conda environment fmripreproc. Needs numpy, pandas, nilearn
"""

import subprocess, os

### SETTABLE PARAMETERS ###
FMRIPOP_FILE='D:/FORSTORAGE/ToolboxesOther/fmripop/post_fmriprep.py'
FOLDER="D:/FORSTORAGE/Data/Project_PCNS/Data_analysis/batch3"
SUBS=['006','010']
TASKS=['cface','movie','ff1','facegng']
TRS=[0.8,0.8,0.8,0.82]
CONFOUND_SUFFIXES=['WithPhysio','WithPhysio','WithPhysio','']
### ###

for SUB in SUBS:
    for ntask in range(len(TASKS)):
        TASK=TASKS[ntask]
        TR=TRS[ntask]
        CONFOUND_SUFFIX=CONFOUND_SUFFIXES[ntask]
        PREFIX=f'{FOLDER}/derivatives/fmriprep/sub-{SUB}/func/sub-{SUB}_task-{TASK}_run-1_'
        NIIPATH=f'{PREFIX}space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        MASKPATH=f'{PREFIX}space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        TSVPATH=f'{PREFIX}desc-confounds_timeseries{CONFOUND_SUFFIX}.tsv'
        
        if os.path.exists(NIIPATH):
            command=f'python {FMRIPOP_FILE} --niipath {NIIPATH} --maskpath {MASKPATH} --tsvpath {TSVPATH} --tr {TR} --fmw_disp_th None --fwhm 8.0 8.0 8.0 --low_pass None --high_pass None --detrend --add_orig_mean_img'
            print(f'Running {SUB}, {TASK}')
            print(command)
            subprocess.run(command)
        else:
            print(f'{SUB} {TASK} not found')