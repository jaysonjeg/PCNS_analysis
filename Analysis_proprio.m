%{
  Superseded by aproprio.py
%}

top_folder = fullfile('D:\FORSTORAGE','Data','Project_PCNS','Data_raw');

mat_files = dir(fullfile(top_folder,'PCNS_*_BL','beh','proprio*','proprio*.mat'));

subjects={};
for i=1:length(mat_files)
    subjects{i} = mat_files(i).name(9:11);
end


%Script to analyse output of ProprioSpace.py and ProprioScroll.py
data=load('proprio_015_Ta___2022_Apr_07_1316_out.mat');
data1=data.data; %4D: blocks x trials x (stimface, ptface) x AU intensities for a single frame from OpenFace
data1=data1(2:end,:,:,:);
delays=data.delays;
nblocks=size(data1,1); 
ourAUs=[5,9]; %only plot these ones
%data1=data1(1:2,:,:,:);
%nblocks=2;

ntrials=size(data1,2);

aulabels={1,'AU01';2,'AU02';3,'AU04';4,'AU05';5,'AU06';6,'AU07';7,'AU09';8,'AU10';9,'AU12';10,'AU14';11,'AU15'
    ;12,'AU17';13,'AU20';14,'AU23';15,'AU25';16,'AU26';17,'AU45'};
data2=reshape(data1,nblocks*ntrials,size(data1,3),size(data1,4)); %collapse blocks and trials into one dimension
n=size(data2,1);
data3=squeeze(data2(:,2,:)-data2(:,1,:)); %find ptface-stimface


%%
%{
figure;
for i=1:16 %Plot distribution of (ptface-stimface) across all blocks/trials. Also reports mean difference, and pvalue, for each AU
    subplot(4,4,i);
    stimface=data2(:,1,i);
    ptface=data2(:,2,i);
    [h,p,ci,stats] = ttest2(ptface,stimface);
    diff=ptface-stimface;
    hist(diff);
    if ~isnan(p)
        title(strcat(aulabels(i,2),": ",string(mean(diff)),", ",string(p)));
    end
end

%}
%%

%{
cutoff=20
data2=data2(1:cutoff,:,:); %code to only look at first few trials
data3=data3(1:cutoff,:);
n=cutoff;
%}
%{
figure;
corrRPptface=[]; %store R and P for trend in AU intensities as trial no. increases
corrRPdiff=[]; %store R and P for trend in (ptface-stimface) difference as trial no. increases
 %columns are [correlation R, correlation pvalue, fit slope, fit intercept]
for i=1:16
    [r,p]=corr((1:n)',data2(:,2,i));
    corrRPptface=[corrRPptface;r,p];
    [r,p]=corr((1:n)',data3(:,i));
    corrRPdiff=[corrRPdiff;r,p];
end
%}
%%
figure;
corrRPMB=[]; %nAUs x 4: stores correlations between ptface and stimface, and linear fit parameters
for j=1:length(ourAUs)
    i=ourAUs(j);
    x=data2(:,1,i); %stimface
    y=data2(:,2,i); %ptface
    [r,p]=corr(x,y);
    po=polyfit(x,y,1);
    f = polyval(po,x); 
    corrRPMB=[corrRPMB;r,p,po(1),po(2)];
    subplot(length(ourAUs),1,j);
    plot(x,y,'.',x,f,'-') 
    xlim([0,4]); xlabel('stimface');
    ylim([0,4]); ylabel('ptface');
    if ~isnan(po(1))
        title(sprintf("%s, R:%.2f, p:%.3f,\n m:%.2f, b:%.2f",aulabels{i,2},r,p,po(1),po(2)));
    end
end