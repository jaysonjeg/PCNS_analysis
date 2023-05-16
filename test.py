"""
This script will make a 1 dimensional function with a single 'bump'. 
Then plot the function. 
Then find the discrete fourier transform of the function
Then remove the high frequency components of the function and inverse fourier transform back to time domain
Then find correlation between the original function and the filtered function
"""
import numpy as np
import matplotlib.pyplot as plt

def make_bump_function(N=1000):
    #Make a 1 dimensional function with a single 'bump' - a truncated sine wave
    x = np.linspace(-4*np.pi, 4*np.pi, N)
    y = np.sin(x)
    bump_boundary = [0,np.pi]
    x_outside_boundary = np.where((x<bump_boundary[0]) | (x>bump_boundary[1]))[0]
    y[x_outside_boundary] = 0
    return x,y

N=1000 #number of time points
x,y=make_bump_function(N)
yf = np.fft.rfft(y)
power_spectrum = np.abs(yf)**2
freq = np.fft.rfftfreq(y.shape[-1])
nfreqs = len(freq)

def remove_n_components(n,which_end='high'):
    #remove n fourier components. Return filtered function and correlation with original function
    yf2 = yf.copy()
    if which_end=='high': #remove high freq components
        yf2[-n:] = 0
    elif which_end=='low': #remove low freq components
        yf2[:n] = 0
    y2 = np.fft.irfft(yf2)
    correlation = np.corrcoef(y,y2)[0,1]
    return y2,correlation

#y2,correlation = remove_n_components(10)

max_n=499
ns = list(range(1,max_n+1)) #try removing n_components for each number in this list

results = [remove_n_components(n,which_end='high') for n in ns]
y2s,corrs = zip(*results)

results = [remove_n_components(n,which_end='low') for n in ns]
y2s_remlow,corrs_remlow = zip(*results)

fig,axs = plt.subplots(nrows=2,ncols=2)
axs[0,0].plot(x,y)
axs[0,0].set_title('original function')
axs[0,1].plot(freq,power_spectrum)
axs[0,1].set_title('power spectrum')
axs[0,1].set_yscale('log')
axs[0,1].set_xlabel('frequency')
axs[1,0].plot(ns,corrs,label='remove high-freq first',color='red')
axs[1,0].plot(ns,corrs_remlow,label='remove low-freq first',color='blue')
axs[1,0].legend()
axs[1,0].set_xlim(left=0,right=500)
axs[1,0].set_xlabel(f'number of fourier components removed (/{nfreqs})')
axs[1,0].set_ylabel('correlation with original function')

ns_of_interest = [250,480,490,495]
fig,axs=plt.subplots(nrows=len(ns_of_interest))
for i in range(len(ns_of_interest)):
    n = ns_of_interest[i]
    axs[i].plot(x,y2s[n-1])
    axs[i].set_title(f'{n}/{nfreqs} high-freq components removed. Corr {corrs[n-1]:.4f}')
fig.tight_layout()
fig.suptitle('Reconstruction accuracy')
plt.show()