cd /Volumes/Scratch/Jayson/...

FOLDER=batch4
PTS=010 012

fmriprep-docker $FOLDER $FOLDER/derivatives/fmriprep participant --participant-label $PTS --fs-no-reconall -vv --resource-monitor --mem-mb 45000 --nprocs 10 --fs-license-file /Applications/freesurfer/7.2.0/license.txt

echo $1
