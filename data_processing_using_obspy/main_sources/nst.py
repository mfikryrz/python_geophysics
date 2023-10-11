import numpy as np
from scipy import interpolate
from scipy import stats
#normal score transform

def nst(traces):
    seis = traces
    ns = np.zeros((seis.shape[0], seis.shape[1]))
    for j in np.arange(0,seis.shape[1],1):
        traces = seis[:,j]
        q1 = np.percentile(traces, 25)
        q3 = np.percentile(traces, 75)
        iqr = q3 - q1
        if iqr > 0:
            h = 2*iqr*(len(traces)**(-1/3))
            nbin = int(np.round((np.max(traces) - np.min(traces))/h))
        else:
            nbin = 2
        h = (np.max(traces) - np.min(traces))/(nbin-1)
        p = np.zeros(nbin)
        l = p*1
        l1 = p*1
        l2 = p*1
        for i in np.arange(0, nbin, 1):
            p[i] = len(np.where((traces >= np.min(traces)+i*h) & (traces < np.min(traces)+(i+1)*h))[0])/len(traces)
            l[i] = np.min(traces)+h/2 + i*h
        p[i] +=1/len(traces)
        P = np.hstack((np.zeros(1), p, np.zeros(1)))
        L = np.hstack((np.min(traces)+h/2-1*h, l, np.min(traces)+h/2 + nbin*h))
        cd = np.cumsum(P)
        cd = cd*(stats.norm.cdf(3)-stats.norm.cdf(-3)) + stats.norm.cdf(-3)
        f = interpolate.interp1d(L, cd, kind='linear')
        pz = f(traces)
        ns[:,j] = stats.norm.ppf(pz)
    traces = ns
    return traces