function [y] = myisnan(x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if iscell(x)
    y=myisnan(x{1});
elseif isstring(x) | ischar(x)
    y=0;
else
    y=isnan(x);
end
end

