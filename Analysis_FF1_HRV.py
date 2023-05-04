from glob import glob
import pandas as pd, numpy as np, re
from mysystole.detection import oxi_peaks

SFREQ=100
NEW_SFREQ=1000

folder='pilot_data/PilotA_FF1'
ppgfiles=glob(f"{folder}/*ppg.csv")
print(len(ppgfiles))

p=re.compile('\S+FF1_(\w+)_Task_\S+.csv') #regular expression for ppgfilenames

data={}
for ppgfile in ppgfiles:
    pt=p.match(ppgfile).group(1)
    data[pt]={}
    ppgarray=np.array(pd.read_csv(ppgfile))
    ppgtimes,ppgvalues=ppgarray[:,0],ppgarray[:,1]
    #data[pt]['ppg']=ppgarray

    signal2,peaks=oxi_peaks(ppgvalues,sfreq=SFREQ,new_sfreq=NEW_SFREQ,clipping=False)
    peaks_indices=np.where(peaks)[0] #index of each peak in 1000Hz resampled signal
    bpms=NEW_SFREQ*60/np.diff(peaks_indices)
    RRSD=np.std(1./bpms) #HRV in RR standard deviation
    bpms_mean,bpms_std=np.mean(bpms),np.std(bpms)
    
    for i in range(len(bpms)):
        if (bpms[i] < bpms_mean - 2*bpms_std) or (bpms[i] > bpms_mean + 2*bpms_std):
            bpms[i]=np.nan #remove values outside 3 standard deviations
    peaktimes=[ppgtimes[int(i*SFREQ/NEW_SFREQ)] for i in peaks_indices]

    bpm_validindices=[i for i in range(len(bpms)) if not(np.isnan(bpms[i]))]
    #indices of non-nan bpm values
    bpmsn=[bpms[i] for i in bpm_validindices] #remove nans
    peaktimesn=[peaktimes[i] for i in bpm_validindices]
    #peak times in original 100Hz signal (some rounding error)

    midtimes=[(peaktimesn[i]+peaktimesn[i+1])/2
              for i in range(len(peaktimesn)-1)
              ] #midpoint of each time in peaktimesn

    data[pt]['bpm']=np.column_stack((midtimes,bpmsn))

    assert(0)
    
            


"""
self.signal=self.oxi.recording[-self.SFREQ*(self.PPG_SECONDS+1):]
self.signal2,self.peaks=oxi_peaks(self.signal,sfreq=self.SFREQ,new_sfreq=self.NEW_SFREQ,clipping=False)
self.bpms=self.NEW_SFREQ*60/np.diff(np.where(self.peaks[-self.NEW_SFREQ*self.PPG_SECONDS:])[0])
if not(np.any(self.bpms < self.HRcutoff[0]) or np.any(self.bpms > self.HRcutoff[1])) and len(self.bpms)>=1:
    #if all the RR intervals are in acceptable heart-rate range and there is at least 1 bpm value
    self.haveBPM=1
    if len(self.bpms)>1:
        self.bpm=self.bpms.mean() #mean of each BPM (one for each RR interval)
    elif len(self.bpms)==1:
        self.bpm=self.bpms[0]
    self.targetbpm=self.bpm*self.FBmultiplier
    self.targetRR=60/self.targetbpm #target duration between notes (s)
    if not(self.firstBPM):
        self.currRR=self.targetRR
        self.firstBPM=True
else: #if calculated RR outside accepted range (usually due to movement)
    self.haveBPM=0
    if len(self.bpms)==0:
        print("0 bpms, %i peaks" % sum(self.peaks))
"""
