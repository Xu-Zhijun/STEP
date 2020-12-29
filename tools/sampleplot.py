#! /usr/bin/env python
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import os

color = ['r', 'orange', 'yellow', 'lime', 'cyan', 'b', 'fuchsia']
titles = ['0', '1', '2']
XN = 3
YN = 2
x = np.arange(400)
y = 0*x
fig, axes= plt.subplots(YN, XN, sharex = True) #gridspec_kw={'height_ratios':[1,3], 'width_ratios':[10, 1]}
#fig.subplots_adjust(left=0.02, bottom=0.02, right=0.99, top=0.93, wspace= 0, hspace= 0)
fig.subplots_adjust(wspace= 0, hspace= 0)
fig.set_size_inches(18, 11)

for k in range(XN):
    for i in range(YN):
        if k != 0:
            axes[i,k].set_yticks([])
        axes[i,k].set_xticks(range(0,400,50))
        axes[i,k].set_xlim(0, len(x))        
        axes[i,k].plot(x, y, '.', markersize=2, color=color[i*XN+k])
        # axes[i,k].set_ylim()
        axes[i,0].set_ylabel('dB', size = 16)
        axes[0,k].set_title(titles[k], size = 16)
plt.suptitle('xxx', y=0.98, size = 20)
# axes[0,0].plot(x, y, '.', markersize=2, color=color[0])
# plt.pause(0.01)
# plt.savefig('test.png')
plt.show()