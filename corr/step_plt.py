#! /usr/bin/env python
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt

def plotinitamp(fftsize, x_n, y_n, intnum):
    fig, axes = plt.subplots(y_n*2, x_n)#, gridspec_kw={'height_ratios':[2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1]})
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.92, wspace= 0, hspace= 0)
    x = np.arange(fftsize)
    y = 0*x
    line1 = [[[0 for _ in range(x_n)] for __ in range(y_n*2)] for ___ in range(intnum)]
    line2 = [[[0 for _ in range(x_n)] for __ in range(y_n*2)] for ___ in range(intnum)]
    backgrounds = [[0 for _ in range(x_n)] for __ in range(y_n*2)]
    for i in range(y_n):
        for k in range(x_n):
            for s in range(intnum):
                line1[s][i*2][k] = axes[i*2, k].plot(x, y, '.', color='blue', markersize=2)[0]
                line2[s][i*2][k] = axes[i*2, k].plot(x, y, '.', color='green', markersize=2)[0]
                line1[s][i*2+1][k] = axes[i*2+1, k].plot(x, y, '-', color='blue')[0]
                line2[s][i*2+1][k] = axes[i*2+1, k].plot(x, y, '-', color='green')[0]
            axes[i*2, k].set_ylim(-200, 200)
            axes[i*2+1, k].set_ylim(-10, 30)
            # axes[i*2, k].grid(True, axis = 'x')
            # axes[i*2+1, k].grid(True, axis = 'x')
            # axes[i*2, k].set_xticks([fftsize/2])
            if (k != 0):
                axes[i*2, k].set_yticks([])
                axes[i*2+1, k].set_yticks([])
            else:
                axes[i*2, k].set_yticks([-90, 0, 90])
                axes[i*2+1, k].set_yticks([-10, 0, 10])
            axes[i*2, k].set_xticks([])
            if (i != y_n -1):
                axes[i*2+1, k].set_xticks([])
            axes[y_n -1, k].set_xticks([fftsize/2])  
            axes[0, k].set_title(k+1)
            axes[i*2, 0].set_ylabel(i+1)
            backgrounds[i*2][k] = fig.canvas.copy_from_bbox(axes[i*2, k].bbox)            
            backgrounds[i*2+1][k] = fig.canvas.copy_from_bbox(axes[i*2+1, k].bbox)
    return fig, line1, line2, backgrounds

def plotsum(fftsize, x_n, y_n, intnum, fin):
    fig, axes = plt.subplots(y_n*2, x_n)
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.92, wspace= 0, hspace= 0)
    x = np.arange(fftsize)
    y = 0*x
    for i in range(y_n):
        for k in range(x_n):
            for s in range(intnum):
                axes[i*2, k].plot(x, np.angle(fin[k+ i*x_n, 0], deg=True), '.', color='blue', markersize=2)
                axes[i*2, k].plot(x, np.angle(fin[k+ i*x_n, 1], deg=True), '.', color='green', markersize=2)[0]
                axes[i*2+1, k].plot(x, np.log10(np.abs(fin[k+ i*x_n, 0]))*10, '-', color='blue')[0]
                axes[i*2+1, k].plot(x, np.log10(np.abs(fin[k+ i*x_n, 1]))*10, '-', color='green')[0]
            axes[i*2, k].set_ylim(-200, 200)
            axes[i*2+1, k].set_ylim(-10, 30)
            # axes[i*2, k].set_xlim(-fftsize, fftsize*2)
            # axes[i*2+1, k].set_xlim(-fftsize, fftsize*2)
            # axes[i*2, k].set_xticks([fftsize/2])                    
            if (k != 0):
                axes[i*2, k].set_yticks([])
                axes[i*2+1, k].set_yticks([])
            else:
                axes[i*2, k].set_yticks([-90, 0, 90])
                axes[i*2+1, k].set_yticks([-10, 0, 10])
            axes[i*2, k].set_xticks([])
            if (i != y_n -1):
                axes[i*2+1, k].set_xticks([])
            axes[y_n -1, k].set_xticks([fftsize/2])  
            axes[0, k].set_title(k+1)
            axes[i*2, 0].set_ylabel(i+1)

ANTNUM = 128
INTNUM = 8
X_N = 5
Y_N = 3
FREAVG = 1
PLOT_RAG = 240 #160
PLOT_STR = 0 + PLOT_RAG
PLOT_END = 640 -PLOT_RAG
FINECH = PLOT_END - PLOT_STR
BASELINE = ANTNUM*(ANTNUM+1)//2
PLOT_CNT = BASELINE//(X_N*Y_N)
BASELINE = 15

TSTART = time.time()
FILEN1 = "1247842824_1247842824_11.sub.npy"
FILEN1 = ".1247842824_1247842840_11_processed_2D_32chans.sub.npy"

print("START...", FILEN1)
CRSM = np.zeros((BASELINE, 2, INTNUM, FINECH), dtype=complex)
#### Select BaseLine ####
PLT_ANT = np.array([10, 144, 162, 204, 244], dtype=int)
PLT_ANT = PLT_ANT//2
SEL_BL = np.zeros(BASELINE, dtype=int)
sel = 0
for i in range(PLT_ANT.size):
    for s in range(i, PLT_ANT.size):
        SEL_BL[sel]= int((ANTNUM*2 - PLT_ANT[i])*(PLT_ANT[i] + 1)/2 + (PLT_ANT[s]-PLT_ANT[i]))
        sel += 1
print(SEL_BL)
# exit()
#### Read npy File ####
with open(str(FILEN1),'rb') as fn1:
    # CRSM = np.load(fn1)[SEL_BL, :, :, PLOT_STR: PLOT_END]
    CRSM = (np.load(fn1)[SEL_BL, :, :, :]).reshape(BASELINE, 2, INTNUM, FINECH, 4).mean(axis = 4)

#### Plot dada File ####
print(CRSM.shape)
FIG, LINE1, LINE2, BACKGROUNDS = plotinitamp(FINECH, X_N, Y_N, 1)
x = np.arange(FINECH)
plotsum(FINECH, X_N, Y_N, 1, CRSM.copy().mean(axis=2))
while(1):
    for S in range(INTNUM):
        FIG.suptitle("File:" + FILEN1 + " Antennas:" + str(ANTNUM) + str(X_N*Y_N) + " Baselines:" + str(BASELINE) +
                    "Time:" + str(S+1) + '/' + str(INTNUM), Y=0.98, size=14)
        for K in range(X_N):             
            for I in range(Y_N):
                FIG.canvas.restore_region(BACKGROUNDS[I][K])
                if (K + I*X_N) < BASELINE :
                    LINE1[0][I*2][K].set_ydata(np.angle(CRSM[K+ I*X_N, 0, S], deg=True)+0*x)     
                    LINE2[0][I*2][K].set_ydata(np.angle(CRSM[K+ I*X_N, 1, S], deg=True)+0*x)  
                    LINE1[0][I*2+1][K].set_ydata(np.log10(np.abs(CRSM[K+ I*X_N, 0, S]))*10+0*x)      
                    LINE2[0][I*2+1][K].set_ydata(np.log10(np.abs(CRSM[K+ I*X_N, 1, S]))*10+0*x)    
        plt.pause(0.1)
plt.show()


