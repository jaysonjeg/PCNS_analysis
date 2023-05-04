function [r] = getcol(T,colname,colA,targetA,colB,targetB,colC,targetC)
%This function is similar to VLOOKUP in excel
%Output column 'colname', only for every row where column 'colA' is non-nan and equal to
%'targetstring', and 'colB' is non-nan and equal to 'targetB'
%Inputs: 
%   T is a table
%   colname is desired column where outputs come from
%   colA(optional): which column to check for non-nan and equivalence
%   to targetA
%   targetA (optional): target string to match
%   colB and targetB also optional
%Output: r, contains all non-nan values in the specified column

%every non-specified argument set to nan

if nargin<=7
    targetC=nan;
end
if nargin<=6
    colC=nan;
end
if nargin<=5
    targetB=nan;
end
if nargin<=4
    colB=nan;
end
if nargin<=3
    targetA=nan;
end
if nargin<=2
    colA=nan;
end

r=[];
for row=1:height(T)
    x=eval(['T.',colname,'(row)']); %value on column 'colname', row 'row'
    if ~myisnan(colA)
        colAx=eval(['T.',colA,'(row)']); %value on column 'colA',row 'row'
        if iscell(colAx)
            colAx=string(colAx{1});
        end
    end
    if ~myisnan(colB)
        colBx=eval(['T.',colB,'(row)']);
        if iscell(colBx)
            colBx=string(colBx{1});
        end
    end
    if ~myisnan(colC)
        colCx=eval(['T.',colC,'(row)']);
        if iscell(colCx)
            colCx=string(colCx{1});
        end
    end
    
    %x needs to be non-nan
    % AND 
    %Either (1) colA is nan,, (2) targetA is nan and colAx is non-nan, or (3) colAx equals targetA
    % AND
    % As above, but for B
    if (~myisnan(x) ...
        & (myisnan(colA) | (myisnan(targetA) & ~myisnan(colAx)) | colAx==targetA) ...
        & (myisnan(colB) | (myisnan(targetB) & ~myisnan(colBx)) | colBx==targetB) ...
        & (myisnan(colC) | (myisnan(targetC) & ~myisnan(colCx)) | colCx==targetC) ...
        )
        r=[r;x];
    end
end
end

