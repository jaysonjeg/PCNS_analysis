D:
cd "D:\FORSTORAGE\Data\Project_PCNS\Data_analysis"

set FOLDER=batch6
set PTS=053 064 065 067

fmriprep-docker %FOLDER% %FOLDER%/derivatives/fmriprep participant --participant-label %PTS% --fs-no-reconall -vv --resource-monitor --mem-mb 45000 --nprocs 10 --fs-license-file freesurfer_license.txt

echo 
timeout 2
echo 

timeout 2
echo 
pause
