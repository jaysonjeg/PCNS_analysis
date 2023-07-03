
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pingouin as pg

data_folder = 'D:\\FORSTORAGE\\Data\\Project_PCNS\\intermediates\\CardioceptionPaper-main\\data'
df1 = pd.read_csv(data_folder + '\\Del1_psychophysics.txt')
df2 = pd.read_csv(data_folder + '\\Del2_psychophysics.txt')

HRD1_raw_folder = f'{data_folder}\\raw\\HRD'
HRD2_raw_folder = f'{data_folder}\\raw\\HRD'

def get_HR(filename):
    """
    Open filename which ends in .txt as a pandas Dataframe with first row as column names. Return the mean of values in column 'listenBPM'
    """
    #df = pd.read_csv(filename, sep='\t', header=0)
    df = pd.read_csv(filename)
    return df['listenBPM'].mean()
def get_HR_for_subject(subject,foldername):
    """
    Given a string 'subject', check if there is a subfolder called 'subject' within 'foldername'. If there is not, return np.nan. If there is, look within that subfolder for a file called 'filtered.txt'. Call get_HR on that file and return the result.
    """
    import os
    if subject in os.listdir(foldername):
        subject_folder = os.path.join(foldername,subject)
        if 'filtered.txt' in os.listdir(subject_folder):
            filtered_file = os.path.join(subject_folder,'filtered.txt')
            return get_HR(filtered_file)
        else:
            return np.nan
    else:
        return np.nan
def enter_HR_into_dataframe(df,foldername):
    """
    Given a dataframe with a column called 'subject', enter the HR for each subject into a new column called 'HR'.
    """
    df['HR'] = df['Subject'].apply(lambda x: get_HR_for_subject(x,foldername))
    return df

df1=enter_HR_into_dataframe(df1,HRD1_raw_folder)
df2=enter_HR_into_dataframe(df2,HRD2_raw_folder)
df1['BayesianThreshold_abs'] = df1['BayesianThreshold'].abs()
df2['BayesianThreshold_abs'] = df2['BayesianThreshold'].abs()

df_combined = pd.concat([df1,df2])

sns.set_context('paper')
print('Top row Intero, Bottom row Extero')
for df in [df1,df2,df_combined]:
    fig,axs=plt.subplots(2,4)
    for cond in ['Intero','Extero']:
        ax_row = {'Intero':0,'Extero':1}[cond]
        data = df[df.Modality==cond]
        ax = axs[ax_row,0]
        def plot_scatter(data,ax,x,y):
            sns.scatterplot(ax=ax,data=data,x=x,y=y)
            stats = pg.corr(data[x],data[y],method='spearman')
            ax.set_title(f'r={stats["r"].values[0]:.2f} p={stats["p-val"].values[0]:.2f}')   
        plot_scatter(df[df.Modality==cond],axs[ax_row,0],'HR','BayesianThreshold')
        plot_scatter(df[df.Modality==cond],axs[ax_row,1],'HR','BayesianSlope')
        plot_scatter(df[df.Modality==cond],axs[ax_row,2],'HR','BayesianThreshold_abs')
        plot_scatter(df[df.Modality==cond],axs[ax_row,3],'BayesianSlope','BayesianThreshold')
    fig.tight_layout()


for df in [df1,df2, df_combined]:
    fig,ax=plt.subplots()
    x = df.loc[df.Modality=='Extero','BayesianThreshold'].values  
    y = df.loc[df.Modality=='Intero','BayesianThreshold'].values
    ax.scatter(x,y)
    ax.set_xlabel('Extero_threshold')
    ax.set_ylabel('Intero_threshold')
    stats = pg.corr(x,y,method='spearman')
    ax.set_title(f'r={stats["r"].values[0]:.2f} p={stats["p-val"].values[0]:.2f}')

plt.show(block=False)

