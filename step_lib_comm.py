#! /usr/bin/env python
import os
import sys
import time
import readfil

def readplotini(inifile):
    FILENAME = []
    PlotReady = 0
    PLOTFILE = []
    with open(inifile,'r') as fd:
        all_lines = fd.readlines()
    for i in range(len(all_lines)):
        #### Skip Empty Line ####
        if len(all_lines[i].split()) == 0:
            continue
        #### Skip # Line ####
        elif "#" == all_lines[i].split()[0]:
            continue
        elif 'PlotReady' in all_lines[i]:
            PlotReady = int(all_lines[i].split()[2]) 
        elif 'SearchPath' in all_lines[i]:
            SearchPath = all_lines[i].split()[2]
        elif 'PLOTFILE' in all_lines[i]:
            PLOTFILE = all_lines[i].split()[2]
    for root, _, files in os.walk(SearchPath):
        for fil in files:
            if fil.endswith(".fil"):
                FILENAME.append(os.path.join(root, fil))
    return PlotReady, FILENAME, PLOTFILE

def readini(inifile):
    PlotTime = []
    Plotrange = 0
    PlotDM = 0.0
    WINDOWSIZE = 250
    RFITHR = 4.0
    IGNORE = []
    CHOFF_LOW = 0
    CHOFF_HIGH = 0
    THRESH = 1.0
    NSMAX = 1
    LODM = 0.0
    HIDM = 1.0
    DDM = 0.1
    with open(inifile,'r') as fd:
        all_lines = fd.readlines()
    for i in range(len(all_lines)):
        #### Skip Empty Line ####
        if len(all_lines[i].split()) == 0:
            continue
        #### Skip # Line ####
        elif "#" == all_lines[i].split()[0]:
            continue
        elif 'THRESH' in all_lines[i]:
            THRESH = float(all_lines[i].split()[2])
        elif 'NSMAX' in all_lines[i]:
            NSMAX = int(all_lines[i].split()[2])
        elif 'LODM' in all_lines[i]:
            LODM = float(all_lines[i].split()[2])
        elif 'HIDM' in all_lines[i]:
            HIDM = float(all_lines[i].split()[2])
        elif 'DDM' in all_lines[i]:
            DDM = float(all_lines[i].split()[2])
        elif 'RFITHR' in all_lines[i]:
            RFITHR = float(all_lines[i].split()[2])
        elif 'IGNORE' in all_lines[i]:
            IGNORE = float(all_lines[i].split()[2])
        elif 'WINDOWSIZE' in all_lines[i]:
            WINDOWSIZE = int(all_lines[i].split()[2])
        elif 'CHOFF_LOW' in all_lines[i]:
            CHOFF_LOW = int(all_lines[i].split()[2])
        elif 'CHOFF_HIGH' in all_lines[i]:
            CHOFF_HIGH = int(all_lines[i].split()[2])
        elif 'AVERAGE' in all_lines[i]:
            AVERAGE = int(all_lines[i].split()[2])
        elif 'FREQAVG' in all_lines[i]:
            FREQAVG = int(all_lines[i].split()[2])
        elif 'PlotTime' in all_lines[i]:
            for s in range(len(all_lines[i].split()) - 2):
                PlotTime.append(all_lines[i].split()[2+s]) 
        elif 'Plotrange' in all_lines[i]:
            Plotrange = float(all_lines[i].split()[2]) 
        elif 'PlotDM' in all_lines[i]:
            PlotDM = float(all_lines[i].split()[2]) 
        elif 'PlotBoxcar' in all_lines[i]:
            PlotBoxcar = float(all_lines[i].split()[2])       
    if (FREQAVG == 0 or AVERAGE == 0) :
        print("AVERAGE or FREQAVG can't be Zero !!!")
        exit()
    return (THRESH, NSMAX, LODM, HIDM, DDM, RFITHR, IGNORE, WINDOWSIZE, CHOFF_LOW, 
            CHOFF_HIGH, PlotBoxcar, PlotTime, Plotrange, PlotDM, AVERAGE, FREQAVG)


def convolve(dn, boxcar):
    conv = dn.copy()
    for i in range(1, boxcar):
        conv[i:] += dn[:-i]
    return conv

# def disbar(max, dn):
#     jd = '\r %2d%% [%s%s]'
#     a = '*'* np.ceil(dn*100/max)
#     b = ' '* ((max-dn)*100//max)
#     c = (dn/max)*100+1
#     print(jd % (c,a,b), end="", flush=True)