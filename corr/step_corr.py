#! /usr/bin/env python
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

ANTNUM = 128
SAMPLE = 51200
SMPNUM = 200
INTNUM = 20
FINECH = 5120*2 #128
BLOCKSIZE = SAMPLE*2*ANTNUM*2*SMPNUM
HEADSIZE = 4096 + SAMPLE*2*ANTNUM*2

BLOCKNUM = 8
DATAHEAD = 4096

CHOUT = 640
OUT_STR = FINECH//128*(64+56) - CHOUT//2
OUT_END = FINECH//128*(64+56) + CHOUT//2


BASELINE = ANTNUM*(ANTNUM+1)//2
DADASIZE = BLOCKNUM * FINECH * BASELINE * 2 * 4
INTCNT =  (SMPNUM//INTNUM) * (SAMPLE//FINECH) #(SMPNUM//INTNUM) *

print("START...")
tstart = time.time()
# DIR1 = "/data/Her_A_Gold/" #1247842824_1247842832_11.sub
FILEN1 = "/data/Her_A_Gold/1247842824_1247842840_11.sub"

# DADA1 = "/data/Her_A_Gold/1247842824_1247842856_09_10kHz_1s.dada"
# DADA2 = "/data/Her_A_Gold/1247842824_1247842856_11_10kHz_1s.dada"

# FILENAME = []
# for root, _, files in os.walk(DIR1):
#     for sub in files:
#         if sub.endswith(".sub") and "_11" in sub:
#             FILENAME.append(os.path.join(root, sub))
# for FILEN1 in FILENAME:
print("Start:", FILEN1)
_, RST_FILEN = os.path.split(FILEN1)
DIN0 = np.zeros((INTNUM, SMPNUM//INTNUM*ANTNUM*2*SAMPLE//FINECH, 
        FINECH), dtype=np.complex64)
CRSM = cp.zeros((BASELINE, 2, CHOUT))
FFT1 = cp.zeros((SMPNUM//INTNUM, ANTNUM, 2, SAMPLE//FINECH, CHOUT))
#### Read sub File ####
with open(str(FILEN1),'rb') as fn1:
    fn1.seek(HEADSIZE)
    RAW1 = np.fromfile(fn1, dtype=np.int8, count = BLOCKSIZE).reshape(
            SMPNUM*ANTNUM*2*SAMPLE, 2)
print("Read sub File %.2f sec"%(time.time() - tstart))
DIN0 =  (RAW1[:, 0] + RAW1[:, 1]*1j).reshape(INTNUM, 
        SMPNUM//INTNUM*ANTNUM*2*SAMPLE//FINECH, FINECH)
print("Raw to Complex %.2f sec"%(time.time() - tstart))
#### Raw to Complex ####
for t in range(INTNUM):
    DIN1 = cp.array(DIN0[t])
    # print("Raw to Complex %.2f sec"%(time.time() - tstart))
    #### FFT ####
    FFT1 = cp.fft.fft(DIN1, n=FINECH, norm = "ortho")[:, OUT_STR: OUT_END].reshape(
            SMPNUM//INTNUM, ANTNUM, 2, SAMPLE//FINECH, CHOUT)
    print("FFT %.2f sec"%(time.time() - tstart))
    # #### SHIFT ####
    # FFT1[:, :, :, :, FINECH//2:] = FFT0[:, :, :, :, :FINECH//2]
    # FFT1[:, :, :, :, :FINECH//2] = FFT0[:, :, :, :, FINECH//2:]
    # # FFT1 = FFT1.swapaxes(0, 1)
    # print("SHIFT %.2f sec"%(time.time() - tstart))
    #### Cross Correlate ####
    ss = 0
    for i in range(ANTNUM):
        for ii in range(i, ANTNUM):
            CRSM[ss, 0] = (FFT1[:, i, 0, :, :] * cp.conj(FFT1[:, ii, 0, :, :])).mean(axis = (0, 1))
            CRSM[ss, 1] = (FFT1[:, i, 1, :, :] * cp.conj(FFT1[:, ii, 1, :, :])).mean(axis = (0, 1))
            ss += 1 
    print("%d Cross Correlate %.2f sec"%(t, time.time() - tstart))
    # CRSMCPU[:, :, t, :] = cp.asnumpy(CRSM[:, :, :, 0].cpu()) + np.array(CRSM[:, :, :, 1].cpu())*1j 
FFT1 = []  
CRSM = []
cp.save(RST_FILEN+".npy", CRSM)
print("SAVE npy File")


