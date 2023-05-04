%{
Get fMRI physiological data into a form usable by fmripop
Scripted by Jayson Jeganathan

Inputs: Resp and cardiac physio data from Siemens scanner as DICOM
Requirements:extractCMRRphysio.m, PhysIO toolbox
%}

%Settable parameters
topfolder='D:\FORSTORAGE\Data\Project_PCNS\Data_analysis\batch6';
links=containers.Map({'CFACE1','MOVIEDI','FF1'},{'cface','movie','ff1'}); %Mapping from task names in sourcedata fmri folder, to task names outside
subs={'053','064','065','067'};
TR=0.8;

%Preparation
keys=links.keys();
assert(exist(fullfile(topfolder,'derivatives')))
folder=fullfile(topfolder,'derivatives','physio');
if ~exist(folder)
    mkdir(folder)
end

for n_sub=1:length(subs)
    sub=subs{n_sub};
    folder_sub=fullfile(folder,sub);
    if ~exist(folder_sub)
        mkdir(folder_sub)
    end
    for n_task=1:length(keys) %For each task in each subject
        taskname_old=keys{n_task};
        taskname_new=links(taskname_old);
        sprintf('Sub %s, task %s',sub,taskname_new)
        physiodicom=getfile(fullfile(topfolder,'sourcedata',sub,['*',taskname_old,'*PHYSIOLOG','*'],'*.IMA'));
        if physiodicom
            
            folder_sub_task=fullfile(folder_sub,taskname_new);
            if ~exist(folder_sub_task)
                mkdir(folder_sub_task)
            end
            %Step 0: Prepare save folders
            savefolder1=fullfile(folder_sub_task,'extractCMRRPhysio_out');
            if ~exist(savefolder1)
                mkdir(savefolder1)
            end
            savefolder2=fullfile(folder_sub_task,'physio_out');
            if ~exist(savefolder2)
                mkdir(savefolder2);
            end

            %Step 1: Use extractCMRRPhysio.m to convert physiological data
            %DICOMs to _Info.log, _PULS.log and _RESP.log. This file is
            %located in ToolboxesMatlab/MB-master
            extractCMRRPhysio(physiodicom,savefolder1); 


            %{
            Step 2: Use PhysIO to convert into format usable by fmripop. Adapted from
            PhysIO/Examples/Siemens_VD/PPU3T
            %}

            info_file=getfile(fullfile(savefolder1,'*_Info.log'));
            c_file=getfile(fullfile(savefolder1,'*_PULS.log'));
            r_file=getfile(fullfile(savefolder1,'*_RESP.log'));

            %From Info File, get NumSlices and NumVolumes
            fid = fopen(info_file);
            while 1
                x=fgets(fid);
                if strcmp(x(1:9),'NumSlices')
                    NSLICES=str2num(x(15:end));
                    assert(NSLICES==60);
                elseif strcmp(x(1:10),'NumVolumes')
                    NSCANS=str2num(x(15:end));
                    break
                end
            end
            fclose(fid);

            physio = tapas_physio_new();
            physio.save_dir = {savefolder2};
            physio.log_files.vendor = 'Siemens_Tics'; %same as PhysIO/Siemens_VD/PPU3T
            physio.log_files.cardiac = {c_file};
            physio.log_files.respiration = {r_file};
            physio.log_files.scan_timing = {info_file};
            physio.log_files.relative_start_acquisition = 0;
            physio.log_files.align_scan = 'last';
            physio.scan_timing.sqpar.Nslices = NSLICES; 
            physio.scan_timing.sqpar.TR = TR;
            physio.scan_timing.sqpar.Ndummies = 0;
            physio.scan_timing.sqpar.Nscans = NSCANS;
            physio.scan_timing.sqpar.onset_slice = 1;
            physio.scan_timing.sync.method = 'scan_timing_log';
            physio.preproc.cardiac.modality = 'PPU';
            physio.preproc.cardiac.filter.include = false;
            physio.preproc.cardiac.filter.type = 'butter';
            physio.preproc.cardiac.filter.passband = [0.3 9];
            physio.preproc.cardiac.initial_cpulse_select.method = 'auto_matched';
            physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 120; %default 90. Increase for tachycardic pt
            physio.preproc.cardiac.initial_cpulse_select.file = 'initial_cpulse_kRpeakfile.mat';
            physio.preproc.cardiac.initial_cpulse_select.min = 0.4;
            physio.preproc.cardiac.posthoc_cpulse_select.method = 'off';
            physio.preproc.cardiac.posthoc_cpulse_select.percentile = 80;
            physio.preproc.cardiac.posthoc_cpulse_select.upper_thresh = 60;
            physio.preproc.cardiac.posthoc_cpulse_select.lower_thresh = 60;
            physio.model.orthogonalise = 'none';
            physio.model.censor_unreliable_recording_intervals = false;
            physio.model.output_multiple_regressors = 'multiple_regressors.txt';
            physio.model.output_physio = 'physio.mat';
            physio.model.retroicor.include = true;
            physio.model.retroicor.order.c = 3;
            physio.model.retroicor.order.r = 4;
            physio.model.retroicor.order.cr = 1;
            physio.model.rvt.include = false;
            physio.model.rvt.delays = 0;
            physio.model.hrv.include = false;
            physio.model.hrv.delays = 0;
            physio.model.noise_rois.include = false;
            physio.model.noise_rois.thresholds = 0.9;
            physio.model.noise_rois.n_voxel_crop = 0;
            physio.model.noise_rois.n_components = 1;
            physio.model.noise_rois.force_coregister = 1;
            physio.model.movement.include = false;
            physio.model.movement.order = 6;
            physio.model.movement.censoring_threshold = 0.5;
            physio.model.movement.censoring_method = 'FD';
            physio.model.other.include = false;
            physio.verbose.level = 2;
            physio.verbose.process_log = cell(0, 1);
            physio.verbose.fig_handles = zeros(1, 0);
            physio.verbose.fig_output_file = 'physio.fig';
            physio.verbose.use_tabs = false;
            physio.verbose.show_figs = false;
            physio.verbose.save_figs = false;
            physio.verbose.close_figs = false;
            physio.ons_secs.c_scaling = 1;
            physio.ons_secs.r_scaling = 1;
            [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(physio);   


            %Step 3: Add physio confounds to confounds time series
            oldtsv=getfile(fullfile(topfolder,['derivatives\fmriprep\sub-',sub],'func',['*task-',taskname_new,'*confounds_timeseries.tsv']));
            s=tdfread(oldtsv,'tab');
            for i=1:size(R,2)
                s=setfield(s,['physio',int2str(i)],R(:,i));
            end
            tdfwrite([oldtsv(1:end-4),'WithPhysio.tsv'],s);
        end
    end
end

beep; pause(2); beep; pause(2); beep;