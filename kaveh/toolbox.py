from scipy.signal import resample
import numpy as np
import scipy.signal
import scipy.fftpack


def resample_to_freq(data, source_frequency, target_frequency):
    """
    resamples the input data from source_frequency to target_frequency
    """
    resample_ratio = float(target_frequency) / source_frequency
    if resample_ratio == 1:
        return data
    else:
        target_n_samples = int(np.size(data) * resample_ratio)
        resampled_data = resample(data, target_n_samples)
        return resampled_data


def closest_argmin(A, B):
    """
    Finds the indices of the nearest value in B to values in A
    Output is the same size as A
    Source: https://stackoverflow.com/a/45350318
    """
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]

def fft_spectrum(data, Fs):
    """
    Returns spectrum as Power/Frequency (dB/Hz)
    """
    yf = scipy.fftpack.fft(data)
    N = data.size
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), N/2)
    power_spectrum = 2.0/(Fs*N) * (np.abs(yf[:N//2])**2) 
    return xf, 10.0*np.log10(power_spectrum)
    

def notch_all_harmonics(signal, base_freq, sampling_rate):
    """
    Stoppass filter at the base_freq and all of its harmonics
    """
    if base_freq > sampling_rate/2.0:
        print('Invalid frequency to notch')
        return
    curr_freq = base_freq
    while curr_freq < sampling_rate/2.0:
        b, a = scipy.signal.iirnotch(curr_freq/(sampling_rate/2.0), 30)
        signal = scipy.signal.lfilter(b, a, signal)
        curr_freq = curr_freq + base_freq
        
    return signal
        

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.lfilter(b, a, data)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.lfilter(b, a, data)
    return y
