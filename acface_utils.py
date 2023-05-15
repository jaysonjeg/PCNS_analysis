import numpy as np

def find_vals_between(start_frames,end_frames,aus):
#For each pairs of values in start_frames and end_frames. Return all rows in aus whose indices are between the lower and upper of these values.
    values=np.empty(len(start_frames),dtype='object')
    for i in range(len(start_frames)):
        start=start_frames[i]
        end=end_frames[i]
        values[i] = aus.iloc[start:end,:]
    return values
def pca_comp0_direction_correct(target_fps,values_resampled,pca):
    #Check the direction of the first principal component. Return whether it increases from trigger onset to middle of stimMove
    mid_stimMove_frame = target_fps * 2 #approximate no of frames from trigger to middle of stimMove
    values_resampled_0_to_mid_stimMove = values_resampled[mid_stimMove_frame,:] - values_resampled[0,:] 
    comp0_0_to_mid_stimMove = values_resampled_0_to_mid_stimMove @ pca.components_[0]
    return comp0_0_to_mid_stimMove > 0 
