"""
Analyse cface1 data
"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sub_path='D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw\\PCNS_067_BL\\beh\\cface1_067_Ta_HFbR_MidHA_2022_Oct_13_0915'
file_prefix='cface1_067_Ta_HFbR_MidHA_2022_Oct_13_0915'
data_path='{}\\{}_out.csv'.format(sub_path,file_prefix)
face_path='{}\\OpenFace\\{}_cam_20fps.csv'.format(sub_path,file_prefix)

data=pd.read_csv(data_path)
face=pd.read_csv(face_path)

frame=np.asarray(face['frame'])
au=np.asarray(face[' AU12_r']) 

triggerframesHA=data['trigger_camNframe'][data['ptemot']=='HA']
triggerframesAN=data['trigger_camNframe'][data['ptemot']=='AN']

#Plot AU12 time series
fig,ax=plt.subplots()
ax.plot(frame,au)
ax.set_ylim(bottom=0)
ax.set_xlim(left=0,right=2000)

#Blue lines are happy trigger. Red lines are angry trigger
for i in triggerframesHA:
    plt.axvline(x=i,color='b')
for i in triggerframesAN:
    plt.axvline(x=i,color='r')

plt.show()