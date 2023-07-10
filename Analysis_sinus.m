%Script to analyse output of PtMirrorsSinusoidalStimulus.py
data=load('sinus_015_Ta___2022_Apr_07_1308_out.mat');
metadata=data.metadata;
ausdata=data.ausdata; %cell array of nblocks x ntrials x getNframes. Each n by 17 AU array
posdata=data.posdata;
tstim=0.04;
%tstim=data.tstim; %time(sec) per stimulus frame presentation

nblocks=size(ausdata,1);
ntrials=size(ausdata,2);
nframes=size(ausdata,3);

allxs=tstim:tstim:nframes*tstim;

%First 2 trials are no feedback, predictable
%Next 5 trials are no feedback, pjump=[0.025,0.03,0.035,0.04,0.045]
%Last 3 trials are feedback present, predictable

a=[]
for i=1:nframes
    datum=ausdata(1,1,i);
    a=[a,size(datum{1},1)];
end
sum(a) %confirm that number of grabbed frames is as expected
%%
block=1;
figure;
auspt=[];

for trial=1:ntrials
    subplot(5,2,trial);
    x=[]; %non-empty x-values
    nonemptydata=[]; %non-empty AU values
    for frame=1:nframes %grab trial 1 AU data (do mean within each frame)
        datum=ausdata{block,trial,frame};
        if ~isempty(datum)
            x=[x;frame*tstim];
            nonemptydata=[nonemptydata;mean(datum,1)]; 
        end
    end
    aus_t1=interp1(x,nonemptydata,allxs,'linear','extrap'); %interpolate frames without AUs
    auspt=cat(3,auspt,aus_t1);
    plot(allxs,[normalize(aus_t1(:,[5,9]),'range',[0,1]),squeeze(normalize(posdata(block,trial,:),'range',[0,1]))]);
end


posdata=squeeze(posdata); %10x500
auspt2=permute(auspt,[3,1,2]); %10x500x17
figure;
for trial=1:ntrials
    subplot(5,2,trial);
    nau=9; %AU12
    thisposdata=posdata(trial,:);
    thisauspt=auspt2(trial,:,nau);

    posphase=(angle(hilbert(thisposdata)));
    auphase=(angle(hilbert(thisauspt)));
    posphaseu=unwrap(posphase);
    auphaseu=unwrap(auphase);

    [~, Ntrials] = size(posphaseu);
    e = exp(1i*(posphaseu - auphaseu));
    plv = abs(sum(e,2)) / Ntrials
    plot([posphase;auphase]')
    %plot([posphaseu;auphaseu]')
    
    diffs=auphase-posphase;
    inds=(diff(sign(posphase))==2); %find indices of sign crossings from neg to positive
    meandiff=mean(diffs(inds)); %mean difference value during sign crossings
    
    title(sprintf("PLV %.2f, meandiff %.2f",plv,meandiff)); %plot PLV and mean difference during sign crossings
end

%%
x=angle(hilbert(thisauspt)); y=angle(hilbert(thisposdata));
diffs=x-y;
figure; plot(diffs); hold on; plot(x); plot(y); hold off