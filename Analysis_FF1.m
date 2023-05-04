format compact
emots={'neutral','happy','angry'};
conds={'T','F'};
subnames={'BP','CF','RW','NK','MC','NB','RT','CM','RP','BK','YD','PR','ED','AB'};

r={}; %no of subjects x [Neutral,Happy,Angry] x [True HR fb, False HR fb]
for i=1:length(subnames)
    subname=subnames{i};
    thisdir=dir(fullfile('data/PilotA_FF1/',['*_',subname,'_Task*out.csv']));
    file=fullfile(thisdir.folder,thisdir.name);
    T=readtable(file);
    
    %get ratings of true HR feedback, feedback working, emotion neutral
    r{i,1,1}=getcol(T,'rating','cond','True','rate_havebpm',1,'emot','NE');
    r{i,1,2}=getcol(T,'rating','cond','False','rate_havebpm',1,'emot','NE');
    
    r{i,2,1}=getcol(T,'rating','cond','True','rate_havebpm',1,'emot','HA');
    r{i,2,2}=getcol(T,'rating','cond','False','rate_havebpm',1,'emot','HA');
    
    r{i,3,1}=getcol(T,'rating','cond','True','rate_havebpm',1,'emot','AN');
    r{i,3,2}=getcol(T,'rating','cond','False','rate_havebpm',1,'emot','AN');
end

rm=cellfun(@mean,r); %display all the means to check that AN/HA are rated higher than NE
rmd=rm(:,:,2)-rm(:,:,1); %for each subject/emotion, (falseFB mean - trueFB mean)

nemot=1; %only looking at NEutral emotion

%Simple model for neutral. T-test of  False-True difference mean for each subject
[h,p]=ttest(rm(:,nemot,1),rm(:,nemot,2));

%Pseudoreplicated model for neutral: 1 + True + False
[h,p]=ttest2(cell2mat(r(:,nemot,1)),cell2mat(r(:,nemot,2)));
%%
Rating=[];
Sub={}; 
Emot={}; %to store data in table
Cond={};
for ncond=1:size(r,3)
    for nemot=1:size(r,2)
        for nsub=1:size(r,1)
            thisLength=length(r{nsub,nemot,ncond})
            Rating=[Rating;r{nsub,nemot,ncond}];
            Sub=[Sub;repmat({subnames{nsub}},thisLength,1)];
            Emot=[Emot;repmat({emots{nemot}},thisLength,1)];
            Cond=[Cond;repmat({conds{ncond}},thisLength,1)];
        end
    end
end
T=table(Rating,Sub,Emot,Cond);
%%
ind_notneutral=find(strcmp(T.Emot,'happy')|strcmp(T.Emot,'angry')); %indices of non-neutral

%Pseudoreplicated model for neutral: 1 + True + False
lme=fitlme(T,'Rating~Cond','Exclude',ind_notneutral);

%LME model of NEutral: 1 + (1|subj)+ True + False
lme=fitlme(T,'Rating~Cond+(1|Sub)','Exclude',ind_notneutral);

