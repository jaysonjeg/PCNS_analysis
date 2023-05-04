
%Put every filename in funcfiles into the IntendedFor field of every file
%in fmapfiles
folder='D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_analysis\\batch6\\'; %MODIFY THIS LINE. The top of the BIDS folder

subfolders=dir([folder,'sub*']); %look in all the folders starting with 'sub-'
subfoldernames={subfolders.name};
dirs={}; inplanedirs={};
for k=1:length(subfolders)
    subfolder=fullfile(subfolders(k).folder,subfolders(k).name);
    fmapfiles=dir([subfolder,'\\fmap\\','*json']);
    for i=1:length(fmapfiles) 
        jsonText=fileread(fullfile(fmapfiles(i).folder,fmapfiles(i).name)); %get the fmap .json
        jsonData=jsondecode(jsonText); 
        dirs{k,i}=jsonData.PhaseEncodingDirection;
        inplanedirs{k,i}=jsonData.InPlanePhaseEncodingDirectionDICOM;
    end
end

%%
%Replace the Intended For in each fmap .json
for k=1:length(subfolders)
    sprintf('k');
    subfolder=fullfile(subfolders(k).folder,subfolders(k).name);
    fmapfiles=dir([subfolder,'\\fmap\\','*json']);
    funcfiles=dir([subfolder,'\\func\\','*bold.nii.gz']);
    funcnames={funcfiles.name};
    for i=1:length(fmapfiles) 
        sprintf('i');
        jsonText=fileread(fullfile(fmapfiles(i).folder,fmapfiles(i).name));
        jsonData=jsondecode(jsonText); 
        jsonData.IntendedFor=[];
        for j=1:length(funcnames)
            jsonData.IntendedFor=[jsonData.IntendedFor, string(['func/',funcfiles(j).name])];
        end
        jsonText2 = jsonencode(jsonData);
        jsonText2 = strrep(jsonText2, ',"', sprintf(',\r"')); %making it more human-readable
        jsonText2 = strrep(jsonText2, '[{', sprintf('[\r{\r'));
        jsonText2 = strrep(jsonText2, '}]', sprintf('\r}\r]'));
        fid = fopen(fullfile(fmapfiles(i).folder,fmapfiles(i).name), 'w');
        fprintf(fid, '%s', jsonText2);
        fclose(fid);

    end
end




