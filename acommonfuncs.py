"""
Contains functions used by different scripts in roject_PCNS/Code_analysis
"""
import numpy as np, pandas as pd, datetime
from glob import glob
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import matplotlib.pyplot as plt
from acommonvars import *

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

def add_columns(t,colnames,dtype=object,value=np.nan):
    for colname in colnames:
        t[colname]=value
        t[colname]=t[colname].astype(dtype)
    return t

def add_table(t,csv_file):
    #Open csv_file as a Pandas dataframe and name it other_t. Combine other_t with dataframe t. For any columns in other_t that are not in t, add them to t. This function assumes that any columns common to t and other_t are already the same.
    other_t = pd.read_csv(f'{temp_folder}\\{csv_file}')
    for col in other_t.columns:
        if ((col not in t.columns) and (col != 'Unnamed: 0')):
            t[col] = other_t[col]
    return t

def str_columns_to_literals(t,columns):
    #Given dataframe t, for all columns in 'columns', their elements will be a string representation of a list. Convert these back into lists
    for column in columns:
        for i in range(t.shape[0]):
            element = t.at[i,column]
            if type(element)==str:
                t.at[i,column] = eval(element)
    return t

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

""" 
def corr(t,group,column_name1, column_name2, robust=True,include_these=None):
    if include_these is None:
        include_these=t.iloc[:,0].copy()
        include_these[:]=True #array of all Trues to include all rows
    if robust: 
        corr_func = stats.spearmanr
    else: 
        corr_func= stats.pearsonr
    x = t.loc[t.use_hrd & include_these & eval(group),column_name1]
    y = t.loc[t.use_hrd & include_these & eval(group),column_name2]
    r,p = corr_func(x,y)
    return r,p
def corr_2groups(t,group1,group2,column_name1,column_name2,robust=True,include_these=None):
    title_string=f'{column_name1} vs {column_name2} '
    if robust: title_string += 'pearsonr:\t'
    else: title_string += 'spearmanr:\t'
    for group in [group1,group2]:
        r,p=corr(t,group,column_name1,column_name2,robust=robust,include_these=include_these)
        title_string += f'{group}: r={r:.2f} p={p:.2f}, '
    print(title_string)   
"""

def scatter(t,groupvar,column1,column2,robust=False,include_these=None,ax=None):
    """
    Scatter plot of column_name1 vs column_name2 from DataFrame t. Scatter points are colored by group. Put correlation within each group on the title. Also plot a line of best fit for each group
    """
    if include_these is None:
        include_these=t.iloc[:,0].copy()
        include_these[:]=True #array of all Trues to include all rows
    if robust: 
        corr_func=stats.spearmanr #could use skipped correlations in pingouin instead
        reg_func = TheilSenRegressor
    else: 
        corr_func= stats.pearsonr
        reg_func = LinearRegression
    if ax is None:
        fig,ax=plt.subplots()
    if robust: title_string = 'Robust: '
    else: title_string = 'Non-robust: '
    groups = [i for i in np.unique(t[groupvar]) if i != '']
    for group in groups:
        x = t.loc[include_these & eval(group),column1]
        y = t.loc[include_these & eval(group),column2]
        r,p = corr_func(x,y)
        xnew = np.linspace(min(x),max(x),100)
        reg = reg_func().fit(x.values.reshape(-1,1),y.values)
        ynew = reg.predict(xnew.reshape(-1,1))
        ax.scatter(x,y,label=group,color=colors[group])
        ax.plot(xnew,ynew,color=colors[group])
        title_string += f'{group}: r={r:.2f} p={p:.2f}, '
    ax.set_xlabel(column1)
    ax.set_ylabel(column2)
    ax.set_title(title_string[:-2])
    ax.legend()
    if ax is None:
        fig.tight_layout()

def scatter_multi(df,groupvar,list_of_col_pairs,dim=(3,4),figsize=(16,8),include_these=None,robust=False,ax=None):
    fig,axs = plt.subplots(*dim,figsize=figsize)
    for i in range(len(list_of_col_pairs)):
        column1,column2 = list_of_col_pairs[i]
        ax = axs[np.unravel_index(i,dim)]
        scatter(df,groupvar,column1,column2,include_these=include_these,robust=robust,ax=ax)
    fig.tight_layout()

def compare(df,groupvar,column,include_these=None,robust=False,to_plot=True,ax=None):
    """
    For each outcome, plot strip-plot of distribution for each clinical group, and display t-test of difference.
    df is a dataframe. group is the column name for grouping variable. 
    Make sure that there are no empty strings '' in the group column
    """
    if include_these is None:
        include_these=t.iloc[:,0].copy()
        include_these[:]=True #array of all Trues to include all rows
    groups = [i for i in np.unique(df[groupvar]) if i != '']
    x = df.loc[df[groupvar]==groups[0],column].dropna()
    y = df.loc[df[groupvar]==groups[1],column].dropna()
    diff = x.mean() - y.mean()
    if robust==True:
        pval = stats.mannwhitneyu(x,y).pvalue
        test_string='MW'
    else:
        pval = stats.ttest_ind(x,y).pvalue
        test_string='ttest'
    if to_plot:
        if ax is None:
            fig,ax=plt.subplots()
        sns.stripplot(ax=ax,data=df.loc[df[groupvar]!='',:],x=groupvar,hue=groupvar,palette=colors,y=column,alpha=0.5)
        ax.set_title(f'diff {diff:.2f} {test_string} p={pval:.2f}')
        ax.set_ylabel(column)
        if ax is None:
            fig.tight_layout()
    else:
        print(f'{column}\t {groups[0]} - {groups[1]}:\t diff={diff:.2f}, {test_string} p={pval:.2f}')

def compare_multi(df,group,columns,dim=(3,4),figsize=(16,8),include_these=None,robust=False,to_plot=True,ax=None):
    """
    For each outcome, plot strip-plot of distribution for each clinical group, and display t-test of difference.
    df is a dataframe. group is the column name for grouping variable. columns is a list of column names in df. dim is a tuple of the dimensions of the subplot grid. 
    Make sure that there are no empty strings '' in the group column
    """
    if to_plot:
        fig,axs = plt.subplots(*dim,figsize=figsize)
    for i in range(len(columns)):
        column=columns[i]
        ax = axs[np.unravel_index(i,dim)]
        compare(df,group,column,include_these=include_these,robust=robust,to_plot=to_plot,ax=ax)
    if to_plot:
        fig.tight_layout()

def pairplot(t,vars=None,x_vars=None,y_vars=None,height=1.5,include_these=None,kind='reg',robust=True,group='group03',title=''):
    """
    Scatterplot of all pairwise variables in vars, and kernel density plots for each variable on the diagonal
    Correlation coefficients and p-values are printed as titles
    """
    if include_these is None:
        include_these=t.iloc[:,0].copy()
        include_these[:]=True #array of all Trues to include all rows
    sns.set_context('talk',font_scale=0.6)
    if vars is not None:
        x_vars=vars
        y_vars=vars
        corner=True
    else:
        corner=False
    grid=sns.pairplot(t.loc[include_these & (t[group]!=''),:],hue=group,corner=corner,kind=kind,x_vars=x_vars,y_vars=y_vars,height=height,palette=colors)
    grid.fig.suptitle(f'Robust={robust}, {title}')
    groups = [i for i in np.unique(t[group]) if i != '']
    #Put correlation values on the off-diagonals
    for i in range(len(x_vars)):
        for j in range(len(y_vars)):
            if (vars is None) or (j>i):
                if robust: corr_func = spearmanr
                else: corr_func= pearsonr
                title=''
                for group in groups:
                    x = t.loc[include_these & eval(group),x_vars[i]]
                    y = t.loc[include_these & eval(group),y_vars[j]]
                    r,p=corr_func(x,y)
                    title += f'{group}: r={r:.2f} p={p:.2f}, '
                grid.axes[j,i].set_title(title)
    #Put differences between groups on the diagonals
    if vars is not None:
        for i in range(len(vars)):
            x = t.loc[include_these & eval(groups[0]),vars[i]]
            y = t.loc[include_these & eval(groups[1]),vars[i]]
            mean_diff = np.mean(x)-np.mean(y)
            p_ttest = stats.ttest_ind(x,y).pvalue
            p_MW = stats.mannwhitneyu(x,y).pvalue
            grid.axes[i,i].set_title(f'{groups[0]}-{groups[1]}={mean_diff:.2f}, ttest p={p_ttest:.2f}, MW p={p_MW:.2f}')
        grid.fig.tight_layout(pad=0,w_pad=0,h_pad=0.5)
    return grid



def get_beh_data(taskname,subject,suffix,use_MRI_task,header='infer'):
    """
    Get behavioural data from *out.csv in 'beh' folder
    suffix is 'out', 'detailed','PPG','face'. Which file to get within the subject's task folder
    Some tasks (cface1, movieDI) have two versions: one for MRI, one for non-MRI.
    if use_MRI_task==True, then get the MRI version, else get the non-MRI version of the task
    header should be set to None if first row of .csv is not column names
    """
    globstring = f"{data_folder}\\PCNS_{subject}_BL\\beh\\{taskname}*Ta_*"
    contents=glob(globstring) #'cface' task non-MRI folder for this particular subject
    if use_MRI_task:
        contents = [i for i in contents if 'Ta_M' in i]
    else:
        contents = [i for i in contents if 'Ta_M' not in i]
    if len(contents) != 1:
        print(f"ERROR: {len(contents)} folders found for {globstring}\n")
        assert(0)
    resultsFolder=contents[0]
    globstring = f"{resultsFolder}\\*{suffix}.csv"
    contents = glob(globstring)
    if len(contents)==1:
        df=pd.read_csv(contents[0],header=header) # make log csv into dataframe
        return df
    if len(contents) != 1:
        print(f"ERROR: {len(contents)} files found for {globstring}\n")
        return None

def get_openface_table(taskname,subject,static_or_dynamic,r_or_c='r'):
    """
    Get the OpenFace intermediates .csv for this subject
    r_or_c: 'r' gets raw action unit values, 'c' gets binary classifications
    """
    globstring = f"{analysis_folder}\\openface_{taskname}\\{subject}"
    contents = glob(globstring)
    if len(contents) != 1:
        print(f"ERROR: {len(contents)} folders found for {globstring}")
        assert(0)
    resultsFolder=contents[0]
    face = pd.read_csv(glob(f"{resultsFolder}\\OpenFace_{static_or_dynamic}\\*_cam_20fps.csv")[0])
    all_frames=np.asarray(face['frame'])
    success=np.array(face[' success'])
    these_aus_labels = [f' {i}_{r_or_c}' for i in aus_labels] #Convert action unit labels into column labels for OpenFace .csv file
    aus = face[these_aus_labels] #get all action units' time series for this subject. The rows are numbered from zero, whereas actual frames are numbered from 1. The error (1/20th of a second) is negligible.
    aus.columns=aus_labels #rename columns, for example from ' AU01_r' to 'AU01'
    return all_frames,aus, success
