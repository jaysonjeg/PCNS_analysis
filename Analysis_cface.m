%Script to analyse output of cface
format compact
folder='D:/FORSTORAGE/Data/Project_PCNS/Data/003/cface1';
face_file=[folder,'/OpenFace/cface1_003_Ta_HFbR_IndHA_2022_Feb_03_1511_cam_30fps.csv'];
out_file=[folder,'/cface1_003_Ta_HFbR_IndHA_2022_Feb_03_1511_out.csv'];
T=readtable(out_file);

%% 
%Convert psychopy data into numeric array cube
UpDown=1; %0 for 'Spacebar', 1 for 'up/down' press (will remove wrong responses)
map_finalemot = containers.Map({'SA', 'HA'}, {0,1});
map_response=containers.Map({'too early', 'just right', 'too late'},{0,1,2}); %too early is 0, just right is 1, too late is 2
str_includeblocks="1:end"; %Settable string (default ":"). Only include these blocks
str_includetrials="1:end"; %Settable string (default ":"). Only include these blocks

names2={'JR','JM','RT','TL'}; 
names={'AB','GJ','NK','DM','RSu'};
names=[names2,names];
names={'JJ','JM','JR','NK','RT','TL'};

ys=[]; covarlists=[]; covarlistnames={};
for nName=1:length(names)
    name=names{nName};
    data=load(['data/Cont_P53_',name,'_Task.mat']);
    data=data.data;
    data=eval("data(:,"+str_includeblocks+","+str_includetrials+");");
    conditions=[];
    for i=1:size(data,1)
        conditions=[conditions,data{i,1,1}.condition];
    end
    conditions=unique(conditions,"stable"); %list of unique conditions, e.g. [0.1,0.9]
    nconditions=length(conditions);
    latestblock=zeros(nconditions,1); 

    nsuperblocks=size(data,1);
    nblocks=size(data,2);
    ntrials=size(data,3);

    y=[]; covarlist1=[]; covar_cond=[]; response=[]; correctresponse=[]; 
 
    for superblock=1:nsuperblocks
        for block=1:nblocks
            for trial=1:ntrials
                datum=data{superblock,block,trial};
                y=[y;double(datum.frameOnKey)];
                response=[response;map_response(datum.response)];  %too early is 0, just right is 1, too late is 2
                correctresponse=[correctresponse;datum.key==datum.finalemot]; %1 if correct button pressed, 0 otherwise
                covarlist1=[covarlist1;[trial,double(datum.ptemot),double(datum.finalemot)]];
                covar_cond(end+1,1)=datum.condition;
            end
        end
    end
    covarlist2=[covar_cond,covarlist1];
    covarlist3=covarlist2(correctresponse==1 & response==1,:); %remove responses which are too early, too late, and incorrect button
    y2=y(correctresponse==1 & response ==1);
    
    ys=[ys;y2]; covarlists=[covarlists;covarlist3]; %these include all subjects 
    covarlistnames=[covarlistnames;repmat({name},length(y2),1)];
end

%%
covarlistss=mat2cell(covarlists,size(covarlists,1),repmat(1,size(covarlists,2),1));
covarlistss{end+1}=covarlistnames;
covarlistss=covarlistss([3,4,1,5,2]);
%%
myvarnames={'ptemot','stimemot','condition','subject','trial no'};
vars=[1,2,3,4];
%[p,tbl2,stats,terms] = anovan(ys,covarlistss(:,vars),'Continuous',length(vars),'model',3,'varnames',myvarnames(vars));
[p,tbl2,stats,terms] = anovan(ys,covarlistss(:,vars),'model',3,'varnames',myvarnames(vars));
figure; results = multcompare(stats,'Dimension',[1,2,3]);
%%
RT=ys; clear ys;
ptemot=covarlistss{1}; stimemot=covarlistss{2};
condition=covarlistss{3}; subject=covarlistss{4}; trialno=covarlistss{5};
T=table(RT,ptemot,stimemot,condition,subject);
%%
lme=fitlme(T,'RT~ptemot*stimemot*condition+(1|subject)');