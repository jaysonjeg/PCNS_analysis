function out = getfile(filepath)
    %e.g. fullfile('C:\Users\*biol.csv') will return the only file matching
    %that
    files=dir(filepath);
    if length(files)==0
        sprintf('No matching files')
        out=false;
    elseif length(files)==1
        out = fullfile(files.folder,files.name);
    else
        sprintf('More than 1 many matching files')
        out=false;
    end
end