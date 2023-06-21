"""
This script will make a 1 dimensional function with a single 'bump'. 
Then plot the function. 
Then find the discrete fourier transform of the function
Then remove the high frequency components of the function and inverse fourier transform back to time domain
Then find correlation between the original function and the filtered function
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def makebump(x,width=1):
    #Make a 1 dimensional function with a single 'bump' - a truncated sine wave
    midpoint = np.median(x)
    y = np.sin((x-midpoint)*np.pi/(width))
    bump_boundary = [midpoint,midpoint+width]
    x_outside_boundary = np.where((x<bump_boundary[0]) | (x>bump_boundary[1]))[0]
    y[x_outside_boundary] = 0
    return y
def makesquare(x,xstart,xend,height):
    #Make 1 dimensional function with 'height' from xstart to xend, and zero elsewhere
    y = np.array([0]*len(x))
    y[np.where((x>=xstart) & (x<=xend))[0]] = height
    return y
def smoothed_noise(x,noise_level=0.1,window=0.5):
    """
    Make a white noise signal y_original with noise_level. Then smooth it with a moving average kernel with window_size
    """
    y_original = np.random.normal(0,noise_level,len(x))
    size = window*len(x)/(max(x)-min(x))
    #size = window*len(x)
    y = scipy.ndimage.uniform_filter1d(y_original,size=int(size))
    return y

def cdf(values):
    #make cumulative distribution function of values, normalized to max of 1
    cdf = np.cumsum(values)
    cdf = cdf/cdf[-1]
    return cdf
def spin(data,n):
    newdata = np.roll(data,n)
    return newdata
def remove_n_components(y,yf,n,which_end='high'):
    #remove n fourier components from  yf. Return filtered function y2 and correlation with original function y
    yf2 = yf.copy()
    if which_end=='high': #remove high freq components
        yf2[-n:] = 0
    elif which_end=='low': #remove low freq components
        yf2[:n] = 0
    y2 = np.fft.irfft(yf2)
    correlation = np.corrcoef(y,y2)[0,1]
    return y2,correlation
def remove_components(y,yf,ns,which_end='high'):
    results = [remove_n_components(y,yf,n,which_end=which_end) for n in ns]
    y2s,corrs_rem = zip(*results)    
    return y2s,corrs_rem
def getcorrs(y,yf,ns):
    #Given fft of function y (yf) and how many components to remove at a time (ns), return correlations between original function and reconstructed function
    _,corrs_remhigh = remove_components(y,yf,ns,'high')
    return corrs_remhigh[::-1]
def autocorr(x):
    #return autocorrelation of x
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]
def variogram(y,max_lag=1000):
    #only calculate variogram up to max_lag_percent of the length of y

    lag_distances = [i for i in range(max_lag)]
    variogram = np.zeros(len(lag_distances))
    for i, lag in enumerate(lag_distances):
        differences = []
        for j in range(1000 - lag):
            differences.append((y[j] - y[j + lag]) ** 2)
            #differences.append(np.var([y[j],y[j+lag]]))
        variogram[i] = np.mean(differences)
        #variogram[i] = np.mean(differences)
    return variogram

def shuffle(x):
    #return a shuffled data vector
    x2 = x.copy()
    np.random.shuffle(x2)
    return x2
def expdecay(length=10):
    #exponentially decaying kernel, with characteristic length (1/decay rate), truncated at 1/epsilon
    epsilon = int(1/length) #truncate at the characteristic length
    expfunc = np.exp(-epsilon*np.arange(length)) #truncate at the characteristic length
    kernel_x = list(np.arange(-length+1,length)) #kernel indices
    kernel_y = list(expfunc[::-1]) + list(expfunc[1:])
    kernel_y = [i/sum(kernel_y) for i in kernel_y]
    return kernel_x,kernel_y
def smooth_exp(data,length=10):
    #smooth data vector x with exponentially decaying kernel
    _,kernel_y = expdecay(length)
    smoothed_data = np.convolve(data,kernel_y,mode='same')
    return smoothed_data
def regress(x,y):
    #linear regression of y1 onto y2. Return the sum of squared errors and the coefficients
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x.reshape(-1,1),y)
    y_pred = reg.predict(x.reshape(-1,1))
    residuals = y - y_pred
    SSE = sum(residuals**2)
    return SSE,reg.coef_,reg.intercept_

N=1000 #number of time points
x = np.linspace(0,8,N)

plot_random=True #fourier phase randomization
do_vario=False
lengths = [1/8,1/4,1/2,1,2,3,5,10,20,25,30,35,40,50,60,70,90] #characteristic lengths to try for exponential decay smoothing
max_lag=1000 #max lag for variogram
n_iters = 10

#y = makebump(x,width=0.6)
y = makesquare(x,1,2,5) + makesquare(x,2.5,5,1) + makesquare(x,7,7.5,7)
#y = smoothed_noise(x,noise_level=0.1,window=0.5)

y = (y-np.mean(y))/np.std(y)

yf = np.fft.rfft(y)
ypower = np.abs(yf)**2
freq = np.fft.rfftfreq(y.shape[-1])
nfreqs = len(freq)
ypower_cdf = cdf(ypower)

max_n=500
ns = list(range(1,max_n+1)) #try removing n_components for each number in this list

y2s_remhigh,corrs_remhigh = remove_components(y,yf,ns,'high')
y2s_remlow,corrs_remlow = remove_components(y,yf,ns,'low')
y_corrs = getcorrs(y,yf,ns)  #correlation with original function y, when reconstructed with increasing numbers of its own fourier components (yf) (starting from low-freq)

y_vario_total = variogram(y,max_lag=1000)

if do_vario:
    y_vario = variogram(y,max_lag=max_lag)
    sample_kernel = expdecay(10)
    y_shuff = shuffle(y)

    y_current = y_shuff
    for j in range(n_iters):
        y_nulls = [smooth_exp(y_current,length=i) for i in lengths]
        y_nulls_varios = [variogram(i,max_lag=max_lag) for i in y_nulls]
        results = [regress(i,y_vario) for i in y_nulls_varios]
        SSEs,coefs,intercepts = zip(*results)
        coefs = [i[0] for i in coefs]
        y_nulls_varios_scaled = [y_nulls_varios[i]*coefs[i]+intercepts[i] for i in range(len(lengths))]
        y_nulls_scaled = [ np.sqrt(np.abs(coefs[i])) * y_nulls[i] + np.sqrt(np.abs(intercepts[i])) * np.random.normal(loc=0,scale=1,size=1000)  for i in range(len(lengths)) ]

        best_index = np.argmin(SSEs)
        print(f'Iteration {j}: best length is {lengths[best_index]}')
        ynull = y_nulls_scaled[best_index]
        y_current = ynull

    y_nulls_scaled_varios = [variogram(i,max_lag=max_lag) for i in y_nulls_scaled] #to check if actual variograms of the rescaled nulls is close to what it should be

    ynullf = np.fft.rfft(ynull)
    ynullpower=np.abs(ynullf)**2
    ynullpower_cdf = cdf(ynullpower)
    ynull_corrs = getcorrs(ynull,ynullf,ns)
    ynull_vario = y_nulls_scaled_varios[best_index]

    #Plots of surrogate production using variogram
    fig,axs=plt.subplots(nrows=3,ncols=len(lengths))
    for i in range(len(lengths)):
        ax=axs[0,i]
        ax.plot(y_vario,'b',label='orig')
        ax.plot(y_nulls_varios_scaled[i],'r',label='null')
        ax.set_ylabel('variance')
        ax.set_xlabel('dist')
        if i==best_index:
            ax.set_title(f'k={lengths[i]} (best fit)')
        else:
            ax.set_title(f'k={lengths[i]}')

        ax = axs[1,i]
        ax.plot(y,'b',label='orig',alpha=0.3)
        ax.plot(y_nulls_scaled[i],'r',label='null',alpha=0.3)

        ax = axs[2,i]
        ax.plot(y_vario,'b',label='orig')
        ax.plot(y_nulls_scaled_varios[i],'r',label='null')
        ax.set_ylabel('variance')
        ax.set_xlabel('dist')
    fig.suptitle('Row 1: new varios. Row 2: new time series. Row 3: actual vario of rescaled nulls')
    fig.tight_layout()

    ynull_vario_total = variogram(ynull,max_lag=1000)

    plot_these=['orig','null']
else:
    plot_these=['orig']
    ynull_corrs,ynullpower,ynullpower_cdf,ynull_vario_total,ynull = 0,0,0,0,0

funcs={'orig':y,'null':ynull}
varios = {'orig':y_vario_total,'null':ynull_vario_total}
cmaps = {'orig':'blue','null':'red','rand':'green'}
all_corrs = {'orig':y_corrs,'null':ynull_corrs}
powers = {'orig':ypower,'null':ynullpower}
powercdfs = {'orig':ypower_cdf, 'null':ynullpower_cdf}


if plot_random: 
    random_phases = np.exp(np.random.uniform(0,np.pi,int(len(y)/2+1))*1.0j)
    yfr = yf*random_phases
    yr = np.fft.irfft(yfr)
    yrf=np.fft.rfft(yr)
    yrpower=np.abs(yrf)**2
    yrpower_cdf = cdf(yrpower)
    yr_corrs = getcorrs(yr,yrf,ns)
    yr_vario_total=variogram(yr,max_lag=1000)
    funcs['rand'] = yr
    varios['rand'] = yr_vario_total
    all_corrs['rand'] = yr_corrs
    powers['rand'] = yrpower
    powercdfs['rand'] = yrpower_cdf
    plot_these.append('rand')


def plotfunc(ax,x,labels):
    for label in labels:
        ax.plot(x,funcs[label],color=cmaps[label],label=label,alpha=0.3)
    ax.set_title('orig func')
def plotpower(ax,freq,labels):
    for label in labels:
        power_spectrum = powers[label]
        ax.plot(freq[1:],power_spectrum[1:],label=label,color=cmaps[label],alpha=0.3)
    ax.set_title('power spectrum')
    ax.set_yscale('log')
    ax.set_xlabel('frequency')
def plothighlow(ax,ns,corrs_remhigh,corrs_remlow):
    ax.plot(ns,corrs_remhigh,label='rem high-freq first',color='pink')
    ax.plot(ns,corrs_remlow,label='rem low-freq first',color='black')
    #ax.legend()
    ax.set_xlim(left=0,right=500)
    ax.set_xlabel(f'remove fourier comps (/{nfreqs})')
    ax.set_ylabel('corr with orig func')
def plotpowercorrs(ax,labels):
    for label in labels:
        corrs = all_corrs[label]
        power_spectrum_cdf = powercdfs[label]
        ax.scatter(power_spectrum_cdf[:-1],corrs,color=cmaps[label],label=label,alpha=0.1)
    ax.set_xlabel('power spectrum cdf')
    ax.set_ylabel('corr with orig func')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
def plotcorrs(ax,ns,labels):
    for label in labels:
        corrs = all_corrs[label]
        ax.scatter(ns,corrs,label=label,color=cmaps[label],alpha=0.2)
    ax.set_xlabel(f'no of fourier comps (/{nfreqs})')
    ax.set_ylabel('corr with orig func')
    ax.set_xlim(left=0,right=50)
def plotautocorr(ax,x,y):
    ax.plot(x,autocorr(y))
    ax.set_xlabel('lag')
    ax.set_ylabel('autocorrelation')
def plotvario(ax,x,labels,max_lag=max_lag):
    for label in labels:
        this_vario = varios[label]
        ax.plot(x[0:max_lag],this_vario,label=label,color=cmaps[label])
    ax.set_xlabel(f'lags up to {max_lag}/1000')
    ax.set_ylabel('variance')


fig,axs = plt.subplots(nrows=2,ncols=4)
plotfunc(axs[0,0],x,plot_these)
plotpower(axs[1,0],freq,plot_these)
plotcorrs(axs[0,1],ns,plot_these)
plothighlow(axs[1,1],ns,corrs_remhigh,corrs_remlow)
plotpowercorrs(axs[0,2],plot_these)
plotautocorr(axs[0,3],x,y)
plotvario(axs[1,3],x,plot_these,max_lag=1000)
fig.tight_layout()


ns_of_interest = [250,480,490,495]
fig,axs=plt.subplots(nrows=len(ns_of_interest))
for i in range(len(ns_of_interest)):
    n = ns_of_interest[i]
    axs[i].plot(x,y2s_remhigh[n-1])
    axs[i].set_title(f'{n}/{nfreqs} high-freq components removed. Corr {corrs_remhigh[n-1]:.4f}')
fig.tight_layout()
fig.suptitle('Reconstruction accuracy')

plt.show(block=False)