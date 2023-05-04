"""
Generic code using OpenFace to analyse facial videos
e.g. 
D:/FORSTORAGE/OpenFace-master/OpenFace-master/x64/Release/FeatureExtraction.exe –f “C:/Users/Jayson/Pictures/Camera Roll/F01-NE-HA.avi” –au_static –out_dir “C:/Users/Jayson/Pictures/Camera Roll” –aus 
"""

### RUN OPENFACE TO GET .CSV FILES ###

import subprocess
 
#'movieDI_test_Te_F_Ricky_Humpty_2022_Mar_03_1643'

vids_folder='D:\\FORSTORAGE\\Data\\Project_PCNS\\Data_raw\\PCNS_067_BL\\beh\\cface1_067_Ta_HFbR_MidHA_2022_Oct_13_0915'#source videos



vidnames=['cface1_067_Ta_HFbR_MidHA_2022_Oct_13_0915_cam_20fps.avi']
out_folder=f'{vids_folder}\OpenFace' #where to save

#laptop
#openfacefolder='C:/Users/c3343721/Desktop/FaceThings/OpenFace-master' 
#home
openfacefolder='D:/FORSTORAGE/OpenFace-master'

openfacefile=f'{openfacefolder}/OpenFace-master/x64/Release/FeatureExtraction.exe'
commands=[f'{openfacefile} -f {vids_folder}\{vidname} -au_static -out_dir {out_folder} -aus' for vidname in vidnames]


for command in commands:
    subprocess.call(command)


### FROM .CSV FILES, DISPLAY AU TIME SERIES ###
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


fig,ax=plt.subplots()

for i in range(len(vidnames)):        
    vidname=vidnames[i][:-4]
    face_path=f'{out_folder}\{vidname}.csv'
    face=pd.read_csv(face_path)
    ax.plot(np.asarray(face[' timestamp']),np.asarray(face[' AU12_r']),label=vidname)
    
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_xlabel('Time (s)')
ax.legend()
plt.show()
