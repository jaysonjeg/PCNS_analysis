%{
Scripted by Jayson Jeganathan
Start in Project_PCNS/Code_analysis/fmri_analysis
Requires behavioural .csv files to be in subject subfolder within the sourcedata folder

%}
format compact
topfolder='D:\FORSTORAGE\Data\Project_PCNS\Data_analysis';
studyname='batch6';
tasknames={'cface','ff1','facegng'};
TRs=[0.8,0.8,0.82];

folder=fullfile(topfolder,studyname,'derivatives');
spmfolder = fullfile(folder,'fmriprep_THEN_fmripop_THEN_spm');
sourcefolder=fullfile(topfolder,studyname,'sourcedata');
subjectfolders=dir(fullfile(folder,'fmriprep/sub*'));
subjectfolders=subjectfolders([subjectfolders.isdir]); %only count folders containing 'sub*'

%%
for sub=1:length(subjectfolders) %for each subject in fmriprep folder
    subname=subjectfolders(sub).name;
    if ~exist(spmfolder) %make spm directory
        mkdir(spmfolder);
    end
    subfolder=[spmfolder,filesep,subname];
    if ~exist(subfolder)
        mkdir(subfolder);
        mkdir([subfolder,filesep,'func']);
        mkdir([subfolder,filesep,'beh']);
        for j=1:length(tasknames)
            files=dir(fullfile(folder,'fmriprep',subname,'func',['sub*task-',tasknames{j},'*confounds-removed.nii.gz']));
            if length(files)==1
                mkdir([subfolder,filesep,'1stLevel_',tasknames{j}]);
            end
        end
    end 
    fgzs=dir(fullfile(folder,'fmriprep',subname,'func','sub*preproc_bold_confounds-removed.nii.gz')); %get nii.gz in fmriprep folder
    for j=1:length(fgzs) %for each bold file
        fgz=fgzs(j);
        f=gunzip(fullfile(fgz.folder,fgz.name));   
        f=f{1};
        movefile(f,[subfolder,filesep,'func']) %move into subfolder
        %f=fullfile(subfolder,'func',fgz(sub).name(1:end-3));
    end
end


%%
%Get multcond .mat file for cface1.
taskname='cface1';
for sub=1:length(subjectfolders) %from log file, save .mat file for multiple conditions
    subname=subjectfolders(sub).name;
    %Get the log file corresponding to fMRI (has 'Ta_M' in it)
    logfiledir=dir(fullfile(sourcefolder,subname(5:end),'beh',[taskname,'*','Ta_M','*'],[taskname,'*','Ta_M','*','_out.csv']));
    %ensure we found a single unique log file
   
    sprintf('%s cface', subname)
    if length(logfiledir)>1
        sprintf('Too many matching')
    elseif length(logfiledir)==0
        sprintf('None matching')
    else
        sprintf('One match')
        logfile=fullfile(logfiledir.folder,logfiledir.name);
        T=readtable(logfile);
        clear multcond;

        multcond.names={'HAHA','HAAN','ANHA','ANAN','instruct_HA','instruct_AN','cond0.8','trigger_HA','trigger_AN'};
        multcond.onsets{1}=getcol(T,'stimMove_onset','type','HAHA');
        multcond.durations{1}=getcol(T,'stimMove_duration','type','HAHA');
        multcond.onsets{2}=getcol(T,'stimMove_onset','type','HAAN');
        multcond.durations{2}=getcol(T,'stimMove_duration','type','HAAN');
        multcond.onsets{3}=getcol(T,'stimMove_onset','type','ANHA');
        multcond.durations{3}=getcol(T,'stimMove_duration','type','ANHA');
        multcond.onsets{4}=getcol(T,'stimMove_onset','type','ANAN');
        multcond.durations{4}=getcol(T,'stimMove_duration','type','ANAN');    

        multcond.onsets{5}=getcol(T,'instruct_onset','ptemot','HA');
        multcond.durations{5}=getcol(T,'instruct_duration','ptemot','HA');
        multcond.onsets{6}=getcol(T,'instruct_onset','ptemot','AN');
        multcond.durations{6}=getcol(T,'instruct_duration','ptemot','AN');

        multcond.onsets{7}=getcol(T,'stimMove_onset','cond',0.8);
        multcond.durations{7}=getcol(T,'stimMove_duration','cond',0.8);

        multcond.onsets{8}=getcol(T,'trigger_onset','ptemot','HA');
        multcond.durations{8}=0.1;
        multcond.onsets{9}=getcol(T,'trigger_onset','ptemot','AN');
        multcond.durations{9}=0.1;

        names=multcond.names; durations=multcond.durations; onsets=multcond.onsets;
        save(fullfile(spmfolder,subname,'beh',['multcond',taskname,'.mat']),'names','durations','onsets');  
    end
end
%{
cface1 no derivs, 8 regressors
instruct_HA: 0,0,0,0,1
pt_HA: 1,1,0
final_HA: 1,0,1,0
ptxfinal: 1,-1,-1,1
ptHA-finalFE-HA: -1,1
pt_diff: 1,1,-1,-1
final_diff: 1,-1,1,-1
trigger_diff: 0,0,0,0,0,0,0,1,-1
%}

%%
%Get multcond .mat file for facegng. New
taskname='facegng';
for sub=1:length(subjectfolders) %from log file, save .mat file for multiple conditions
    subname=subjectfolders(sub).name;
    logfiledir=dir(fullfile(sourcefolder,subname(5:end),'beh',[taskname,'*','Ta_M','*'],[taskname,'*','Ta_M','*','_out.csv']));
    
    sprintf('%s facegng', subname)
    if length(logfiledir)>1
        sprintf('Too many matching')
    elseif length(logfiledir)==0
        sprintf('None matching')
    else
        sprintf('One match')
        logfile=fullfile(logfiledir.folder,logfiledir.name);
        T=readtable(logfile);
        clear multcond;

        multcond.names={'fear_target','fear_distractor','calm_target','calm_distractor'};
        multcond.onsets{1}=getcol(T,'onset_fixation','realcond','fear target');
        multcond.onsets{2}=getcol(T,'onset_fixation','realcond','fear distractor');
        multcond.onsets{3}=getcol(T,'onset_fixation','realcond','calm target');
        multcond.onsets{4}=getcol(T,'onset_fixation','realcond','calm distractor');
        multcond.durations{1}=0.5;
        multcond.durations{2}=0.5;
        multcond.durations{3}=0.5;
        multcond.durations{4}=0.5;

        names=multcond.names; durations=multcond.durations; onsets=multcond.onsets;
        save(fullfile(spmfolder,subname,'beh',['multcond',taskname,'.mat']),'names','durations','onsets'); 
    end
end

%{
SPM Contrasts for facegng task
target-distractor: 1,-1,1,-1
fear-calm: 1,1,-1,-1
fear_target: 1,0,0,0
calm_distractor: 0,0,0,1
%}

%%
%Get multcond .mat file for FF1. New. Haven't thought much about it.
taskname='FF1';
for sub=1:length(subjectfolders) %from log file, save .mat file for multiple conditions
    subname=subjectfolders(sub).name;
    logfiledir=dir(fullfile(sourcefolder,subname(5:end),'beh',[taskname,'*','Ta_M','*'],[taskname,'*','Ta_M','*','_out.csv']));
    sprintf('%s ff1', subname)
    if length(logfiledir)>1
        sprintf('Too many matching')
    elseif length(logfiledir)==0
        sprintf('None matching')
    else
        sprintf('One match')

        logfile=fullfile(logfiledir.folder,logfiledir.name);
        T=readtable(logfile);
        clear multcond;
        multcond.names={'stimNE','stimHA','stimAN','falsefeedback','rating'}; 
        multcond.onsets{1}=getcol(T,'stim_onset','emot','NE');
        multcond.onsets{2}=getcol(T,'stim_onset','emot','HA');
        multcond.onsets{3}=getcol(T,'stim_onset','emot','AN');
        multcond.durations{1}=getcol(T,'stim_duration','emot','NE');
        multcond.durations{2}=getcol(T,'stim_duration','emot','HA');
        multcond.durations{3}=getcol(T,'stim_duration','emot','AN');

        %Write code for multcond{4}, that is the false feedback
        falsefeedback=[0;diff(getcol(T,'stim_FBmultiplier')~=1)];
        onsets=(falsefeedback==1); 
        offsets=(falsefeedback==-1);
        onset_times=T.stim_onset(onsets);
        offset_times=T.stim_onset(offsets);
        if length(offset_times)==length(onset_times)-1
            offset_times=[offset_times;T.stim_onset(end)];
        end
        assert(length(offset_times)==length(onset_times));
        durations=offset_times-onset_times;
        multcond.onsets{4}=onset_times;
        multcond.durations{4}=durations;

        multcond.onsets{5}=getcol(T,'rate_onset');
        multcond.durations{5}=getcol(T,'rate_duration');

        names=multcond.names; durations=multcond.durations; onsets=multcond.onsets;
        save(fullfile(spmfolder,subname,'beh',['multcond',taskname,'.mat']),'names','durations','onsets'); 
    end
end
%{
rating: 0,0,0,0,1
false feedback: 0,0,0,1,0
stim: 1,1,1,0,0
stim HA > NE: -1,1,0,0,0
%}

%%
%Run SPM for each task/subject
for sub=1:length(subjectfolders)
    subname=subjectfolders(sub).name;
    subfolder=[spmfolder,filesep,subname];
    for j = 1:length(tasknames)
        taskname=tasknames{j};
        TR=TRs(j);
        sprintf('%s %s, TR %.2f', subname,taskname,TR)
        spm('Defaults','fMRI');
        spm_jobman('initcfg');
        clear matlabbatch
        f=dir(fullfile(subfolder,'func',['*',taskname,'*.nii']));
        if length(f)==1 %if it is present
            f=fullfile(f.folder,f.name);
            matlabbatch{1}.spm.stats.fmri_spec.dir = {fullfile(subfolder,['1stLevel_',taskname])};
            matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
            matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR; %0.82 for siemens original, 0.8 for cmrr
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60; %60 or 36
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;
            matlabbatch{1}.spm.stats.fmri_spec.sess.scans=cellstr(f);
            %matlabbatch{1}.spm.stats.fmri_spec.sess.scans=getfuncs(f,1:356);

            multif=dir(fullfile(subfolder,'beh',['*',taskname,'*.mat']));
            multif=fullfile(multif.folder,multif.name);
            matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {multif};
            matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0,0];
            %matlabbatch{1}.spm.stats.fmri_spec.mthresh = -inf;

            matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(subfolder,['1stLevel_',taskname],'SPM.mat')};
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;

            spm_jobman('run',matlabbatch);
            clear matlabbatch
        end
    end
end


