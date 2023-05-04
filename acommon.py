"""
Contains common functions for Project_PCNS/Code_analysis
"""

import numpy as np, pandas as pd
from datetime import datetime

class clock():
    """
    How to use
    c=acommon.clock()
    print(c.time())
    """
    def __init__(self):
        self.start_time=datetime.now()       
    def time(self):
        end_time=datetime.now()
        runtime=end_time-self.start_time
        runtime_sec = runtime.total_seconds()
        return runtime_sec,'{:.1f} sec.'.format(runtime_sec)

def get_redcap(redcap_file = "CogEmotPsych_DATA_2022-10-20_1804.csv"):  
    """
    return redcap table
    """

    redcap_folder = "C:\\Users\\c3343721\\Google Drive\\PhD\\Project_PCNS\\BackupRedcap"
    redcap_folder="C:\\Users\\Jayson\\Google Drive\\PhD\\Project_PCNS\\BackupRedcap"
    
    t=pd.read_csv(f"{redcap_folder}\\{redcap_file}")

    #data.query('pilotorreal==2 & group==2') #another way to get particular data

    healthy=((t.group==1) & (t.pilotorreal==2))
    clinical=((t.group==2) & (t.pilotorreal==2))

    healthy_attended=((healthy) & (t.attended==1))
    clinical_attended=((clinical) & (t.attended==1))

    healthy_didmri=((healthy) & (t.attended_fmri==1))
    clinical_didmri=((clinical) & (t.attended_fmri==1))
    
    return t