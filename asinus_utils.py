import numpy as np, pandas as pd
from scipy import signal,fftpack

time_interval = 0.04 #seconds per frame
fs = int(1/time_interval) #sampling frequency

def lowpass(data,lowcut,axis=-1):
    order = 10
    b, a = signal.butter(order, lowcut, btype='low', fs=fs, output='ba')
    y = signal.lfilter(b, a, data,axis=axis)
    return y    

def PLV(x,y):
    #calculate phase locking value of vectors x and y
    e = np.exp(1j * (x - y))
    return np.abs(np.sum(e)) / len(e)

def get_phase_lag_FFT(x,y):
    """
    Calculate phase lag between two signals x and y using FFT. Calculate the fourier transform of each signal. Returns:
        phase_rad: phase lag in radians for each bin
    """
    xfft = np.fft.fft(x)
    yfft = np.fft.fft(y)
    phase_rad = np.angle(xfft/yfft)
    return phase_rad

def get_peak_heights(ts):
    #Given a time series, return the heights of all peaks. Peaks should be separated by 30 frames
    peaks,properties = signal.find_peaks(ts, distance=30)
    return ts[peaks]

def get_grad_peak_heights(ausn_smoo):
    """
    Returns a list of peak heights of the gradient of the time series in ausn
    ausn is 3D array of ntrials * nframes * nAUs
    lowpass_cutoff is the lowpass cutoff in Hz. try lowpass_cutoff 1 (heavy smoothing), 2 or 4 (light smoothing)
    """
    ausn_smoo_grad = np.gradient(ausn_smoo,axis=1)
    ausn_smoo_grad_peaks = [get_peak_heights(ausn_smoo_grad[i,:]) for i in range(ausn_smoo_grad.shape[0])]
    ausn_smoo_grad_peaks = np.hstack(ausn_smoo_grad_peaks)
    return ausn_smoo_grad_peaks

def power_in_band(data,freqs, lowcut,highcut):
    #Returns power in a frequency band, divided by total power
    fft_data = fftpack.fft(data)
    mask = np.where((freqs >= lowcut) & (freqs <= highcut))
    power_spectrum = np.abs(fft_data)**2
    total_power = np.sum(power_spectrum[0:np.argmax(freqs)])
    return np.sum(power_spectrum[mask])/total_power  