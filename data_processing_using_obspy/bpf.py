import numpy as np
from butterworth import butter_bandpass_filter, butter_lowpass_filter, butter_highpass_filter

def bpf(traces, lo, hi, sr):
    bpftraces = []
    for i in range(traces.shape[1]):
        bpftraces.append(butter_bandpass_filter(traces[:,i],lo,hi,1000/sr,order=5))
    return np.asarray(bpftraces).T

def lpf(traces, lo, sr):
    lpftraces = []
    for i in range(traces.shape[1]):
        lpftraces.append(butter_lowpass_filter(traces[:,i],lo,1000/sr,order=5))
    return np.asarray(lpftraces).T

def hpf(traces, hi, sr):
    hpftraces = []
    for i in range(traces.shape[1]):
        hpftraces.append(butter_highpass_filter(traces[:,i],hi,1000/sr,order=5))
    return np.asarray(hpftraces).T