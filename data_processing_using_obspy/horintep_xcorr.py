import numpy as np
import matplotlib.pyplot as plt

data = np.load('DATA/seismic2d.npy')

print(data.shape)
data = data[5:None,:]
vmin, vmax = np.percentile(data,[5,95])

plt.imshow(data, aspect='auto', cmap='bwr_r',vmin=vmin, vmax=vmax)
plt.show()