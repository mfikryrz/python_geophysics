import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

traces = np.load('seismic2d.npy')
traces = traces[5:None,:]


p = 255
np.save('picked1', p)

w = 100
searchw = 100
snapwindow = 5
traceno = 74
polarity = 'Trough'

vmin,vmax = np.percentile(traces,[5,95])
plt.imshow(traces, aspect='auto', cmap='bwr_r', vmin=vmin, vmax=vmax)

for j in range(10):
    trace1 = traces[:, traceno+j] #reff
    trace2 = traces[:,traceno+j+1]

    p = np.load('picked1.npy')
    search_template_trace1 = trace1[int(p - w):int(p + w)]
    score_index = np.array([])  # 950-1130
    score = np.array([])
    for i in range(2 * (searchw + w) + 1):
        try:
            search_template_trace2 = trace2[int(p - w) + int(-searchw + i):int(p + w) + int(-searchw + i)]
            # corr = np.corrcoef(search_template_trace1, search_template_trace2)[0, 1]  # argmax
            corr = np.dot(search_template_trace1,search_template_trace2)/(norm(search_template_trace1)*norm(search_template_trace2))
            score = np.append(score, corr)
            score_index = np.append(score_index, i)
        except:
            pass


    best_score_index = np.argmax(score)
    best_depth_index = score_index[best_score_index]
    newpick = p + (best_depth_index - searchw)

    try:
        tracewind = trace2[int(newpick - snapwindow):int(newpick + snapwindow)]
        if polarity == 'Trough':
            ptidx = np.argmin(tracewind)
            newpick = (newpick - snapwindow + ptidx)
        elif polarity == 'Peak':
            ptidx = np.argmax(tracewind)
            newpick = (newpick - snapwindow + ptidx)
    except:
        pass

    np.save('picked1', newpick)
    plt.plot(traceno+j, newpick,'k*')
plt.show()