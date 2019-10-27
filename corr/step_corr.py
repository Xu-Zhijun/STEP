#! /usr/bin/env python
import argparse
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import torch

parser=argparse.ArgumentParser(description="Correlator of MWA SUB files")
parser.add_argument("fin", type=str,help="input SUB file")
parser.add_argument('-o', type=str, dest='outdir', help='output directory', default="./")
parser.add_argument("-c", type=int, dest='nChans', help="number of fine channels", default=128)
parser.add_argument("-b", type=int, dest='nBlocs', 
    help="number of input blocks to process", default=200)
parser.add_argument("-i", type=int, dest='nOutB', 
    help="number of output blocks, must be able to divise number of input blocks ", default=8)
parser.add_argument("-m", type=int, dest='Mode', 
    help="0: numpy, 1: pytorch, 2, cupy, 3, benchmark=0~2", default=1)
parser.add_argument("-p", dest='plot', 
    help="plot correlator result", action='store_true')
args=parser.parse_args()

print("Init...")
sys.stdout.flush()
ANTNUM = 128            # Number of antennas XX + YY
DATAHEAD = 4096         # Sub file header size
SAMPLE = 51200          # Number of time sample per 40ms per antennas
SMPNUM = args.nBlocs    # Number of input blocks, default 200
INTNUM = args.nOutB     # Number of output blocks, default 8
FINECH = args.nChans    # Number of fine channels, default 128
BLOCKSIZE = SAMPLE*2*ANTNUM*2*SMPNUM//INTNUM    # Number of data to read
HEADSIZE = 4096 + SAMPLE*2*ANTNUM*2     # Number of data to skip

CHOUT = FINECH
OUT_STR = 0 #FINECH//128*(64+56) - CHOUT//2
OUT_END = FINECH #FINECH//128*(64+56) + CHOUT//2

BASELINE = ANTNUM*(ANTNUM+1)//2
INTCNT =  (SMPNUM//INTNUM) * (SAMPLE//FINECH)
assert INTCNT == int(SMPNUM/INTNUM*SAMPLE/FINECH)

FILEN1 = "/data/Her_A_Gold/1247842824_1247842840_11.sub"
FILEN1 = "/data/Her_A_Gold/1247842824_1247842840_12.sub"
FILEN1 = ".1247842824_1247842840_11_processed_3D_32chans.sub"
FILEN1 = args.fin
# BLOCKNUM = 8
# DADASIZE = BLOCKNUM * FINECH * BASELINE * 2 * 4
# DADA1 = "/data/Her_A_Gold/1247842824_1247842856_09_10kHz_1s.dada"
# DADA2 = "/data/Her_A_Gold/1247842824_1247842856_11_10kHz_1s.dada"

# DIR1 = "/data/Her_A_Gold/" #1247842824_1247842832_11.sub
# FILENAME = []
# for root, _, files in os.walk(DIR1):
#     for sub in files:
#         if sub.endswith(".sub") and "_11" in sub:
#             FILENAME.append(os.path.join(root, sub))
# for FILEN1 in FILENAME:
BENCHMARK = "benchmark."+torch.cuda.get_device_name(0)+".txt"
if args.Mode == 3: 
    LoopNum = 3
else:
    LoopNum = 1
    RunMode = args.Mode

for m in range(LoopNum):
    if args.Mode == 3:
        RunMode = m
    print("Starting...", FILEN1, "Mode:", RunMode)
    sys.stdout.flush()
    tstart = time.time()
    _, RST_FILEN = os.path.split(FILEN1)
    RST_FILEN = RST_FILEN.strip('.sub')

    CRSMCPU = np.zeros((BASELINE, 2, INTNUM, CHOUT), dtype=complex)
    if RunMode == 1:
        cuda = torch.device("cuda")
        CRSM = torch.zeros((BASELINE, 2, CHOUT, 2), dtype=torch.float32, device=cuda)
    elif RunMode == 2:
        CRSM = cp.zeros((BASELINE, 2, CHOUT), dtype=cp.complex64)
    #### Read sub File ####
    with open(str(FILEN1),'rb') as fn1:
        for t in range(INTNUM):
            fn1.seek(HEADSIZE+t*BLOCKSIZE)
            RAW1 = np.fromfile(fn1, dtype=np.int8, count = BLOCKSIZE).reshape(
                    SMPNUM//INTNUM*ANTNUM*2*SAMPLE//FINECH, FINECH, 2)
            # Raw to Complex #
            if RunMode == 0:
                FFT1 = RAW1[:, 0] + 1j*RAW1[:, 1]
                # FFT #
                FFT1 = np.roll(np.fft.fft(FFT1, n=FINECH, norm = "ortho")[:, OUT_STR: OUT_END].reshape(
                        SMPNUM//INTNUM, ANTNUM, 2, SAMPLE//FINECH, CHOUT), FINECH//2, axis=4)
                # print("FFT %.2f sec"%(time.time() - tstart))
                # Cross Correlate #
                ss = 0
                for i in range(ANTNUM):
                    for ii in range(i, ANTNUM):
                        CRSMCPU[ss, 0, t, :] = (FFT1[:, i, 0, :, :] * np.conj(FFT1[:, ii, 0, :, :])).mean(axis = (0, 1))
                        CRSMCPU[ss, 1, t, :] = (FFT1[:, i, 1, :, :] * np.conj(FFT1[:, ii, 1, :, :])).mean(axis = (0, 1))
                        ss += 1

            elif RunMode == 1:
                FFT1 = torch.from_numpy(RAW1).float().cuda()
                # FFT #
                FFT1 = torch.roll(torch.fft(FFT1, 1, normalized = True)[:, OUT_STR: OUT_END, :].view(
                        SMPNUM//INTNUM, ANTNUM, 2, SAMPLE//FINECH, CHOUT, 2), FINECH//2, dims=4)
                # print("FFT %.2f sec"%(time.time() - tstart))
                # Cross Correlate #
                ss = 0
                for i in range(ANTNUM):
                    for ii in range(i, ANTNUM):
                        CRSM[ss, 0, :, 0] = (FFT1[:, i, 0, :, :, 0] * FFT1[:, ii, 0, :, :, 0] +
                                            FFT1[:, i, 0, :, :, 1] * FFT1[:, ii, 0, :, :, 1]).mean(dim = (0, 1))
                        CRSM[ss, 0, :, 1] = (FFT1[:, i, 0, :, :, 1] * FFT1[:, ii, 0, :, :, 0] -
                                            FFT1[:, i, 0, :, :, 0] * FFT1[:, ii, 0, :, :, 1]).mean(dim = (0, 1))
                        CRSM[ss, 1, :, 0] = (FFT1[:, i, 1, :, :, 0] * FFT1[:, ii, 1, :, :, 0] +
                                            FFT1[:, i, 1, :, :, 1] * FFT1[:, ii, 1, :, :, 1]).mean(dim = (0, 1))
                        CRSM[ss, 1, :, 1] = (FFT1[:, i, 1, :, :, 1] * FFT1[:, ii, 1, :, :, 0] -
                                            FFT1[:, i, 1, :, :, 0] * FFT1[:, ii, 1, :, :, 1]).mean(dim = (0, 1))
                        ss += 1
                CRSMCPU[:, :, t, :] = np.array(CRSM[:, :, :, 0].cpu()) + np.array(CRSM[:, :, :, 1].cpu())*1j 

            else:
                FFT1 = cp.asarray(RAW1[:, 0] + 1j*RAW1[:, 1])
                # FFT #
                FFT1 = cp.roll(cp.fft.fft(FFT1, n=FINECH, norm = "ortho")[:, OUT_STR: OUT_END].reshape(
                        SMPNUM//INTNUM, ANTNUM, 2, SAMPLE//FINECH, CHOUT), FINECH//2, axis=4)
                # print("FFT %.2f sec"%(time.time() - tstart))
                # Cross Correlate #
                ss = 0
                for i in range(ANTNUM):
                    for ii in range(i, ANTNUM):
                        CRSM[ss, 0, :] = (FFT1[:, i, 0, :, :] * cp.conj(FFT1[:, ii, 0, :, :])).mean(axis = (0, 1))
                        CRSM[ss, 1, :] = (FFT1[:, i, 1, :, :] * cp.conj(FFT1[:, ii, 1, :, :])).mean(axis = (0, 1))
                        ss += 1
                CRSMCPU[:, :, t, :] = cp.asnumpy(CRSM)

            print("%d Cross Correlate %.2f sec"%(t, time.time() - tstart))
            sys.stdout.flush()
    np.save(RST_FILEN+"."+str(FINECH)+"chs"+".mode"+str(RunMode)+".npy", CRSMCPU)
    print("SAVE npy File", time.time() - tstart)
    with open(BENCHMARK, 'a+') as F:
        print("Saving", BENCHMARK, "Mode: %d, Average Time: %d, Spend Time: %.2f \n"%(
                RunMode, INTNUM, time.time() - tstart))
        F.write("Mode: %d, Average Time: %d, Spend Time: %.2f \n"%(RunMode, INTNUM, time.time() - tstart))


