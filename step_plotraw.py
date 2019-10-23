#! /usr/bin/env python
import os
import time
import numpy as np
import readfil
import step_lib_comm
import step_lib_plt as splt
from matplotlib.backends.backend_pdf import PdfPages

def frbplot(filen, ststart):
    #### Read Config File ####
    (_, NSMAX, _, _, _, tthresh, IGNORE, winsize, choff_low, choff_high, plotpes, plotbc,
        plotime, plotrange, plotDM, average, freqavg) = step_lib_comm.readini("frbcfg.ini")
    choff_high = choff_high//freqavg
    choff_low = choff_low//freqavg
    # print(winsize)
    #### READ HEADER ####
    _, rst_filen = os.path.split(filen)
    ispsrfits = False
    if filen.endswith(".fil"):
        header, headsize = readfil.read_header(filen)
    elif filen.endswith(".fits"):
        header, psrdata = step_lib_comm.read_psrfits(filen, ststart)
        ispsrfits = True
        
    if ispsrfits:
        totalsm = header['totalsm']
    else:
        totalsm = readfil.samples_per_file(filen, header, headsize)
    totalch = header['nchans']
    numbits = header['nbits']
    smptm = header['tsamp']*1e6
    smaple = totalsm//average
    numblock = smaple//winsize
    # print(smaple, totalsm, numblock)
    totalsm = numblock*winsize*average
    smaple = numblock*winsize
    nchan = totalch//freqavg
    smptm *= average
    maxbc = plotbc
    # print(smaple, totalsm, numblock)
    # calc time to samples #
    maxsm = np.zeros(len(plotime))
    for i in range(len(plotime)):
        maxsm[i] = np.round((float(plotime[i])/header['tsamp'] + plotbc//2)/average)
    # print(plotime[0], maxsm[0], float(plotime[0])/(header['tsamp']))
    # Calc DM to Delay #
    if header['foff'] < 0:
        chbwd = - header['foff']*1e-3
        higch = header['fch1']*1e-3 - chbwd*freqavg*choff_high 
        lowch = header['fch1']*1e-3 - chbwd*(totalch - freqavg*choff_low - 0.5)
    else: # Reverse the Frequency if foff > 0
        chbwd = header['foff']*1e-3
        higch = header['fch1']*1e-3 + chbwd*(totalch - freqavg*choff_high - 0.5)
        lowch = header['fch1']*1e-3 + chbwd*freqavg*choff_low
    chfrq = np.arange(higch, lowch, -chbwd*freqavg)
    delay = (4148.741601*plotDM*(chfrq**(-2) - (higch)**(-2))/smptm).round()
    # print(higch, lowch, chbwd)
    #### Read Filterbank File ####
    if ispsrfits:
        data_raw = psrdata.reshape(smaple, average, nchan, freqavg).mean(axis=(1,3))
    elif numbits >= 8:    # BITS NUMBER 8/16/32
        with open(str(filen),'rb') as fn:
            fn.seek(headsize)
            if   numbits == 32:
                fin = np.fromfile(fn, dtype=np.float32, count=totalsm*totalch)
            elif numbits == 16:
                fin = np.fromfile(fn, dtype=np.uint16, count=totalsm*totalch)
            elif numbits == 8:
                fin = np.fromfile(fn, dtype=np.uint8, count=totalsm*totalch)
        data_raw = fin.reshape(smaple, average, nchan, freqavg).mean(axis=(1,3))
        if fin.size != totalsm*totalch:
            print("FILE SIZE ERROR   %s Time:%.2f sec"%(rst_filen, 
                    (time.time() - tstart)))
            exit()
    else:               # BITS NUMBER 1/2/4
        numbtch = 8//numbits
        with open(str(filen),'rb') as fn:
            fn.seek(headsize)
            fin = np.fromfile(fn, dtype=np.uint8, count=totalsm*totalch//numbtch)
        if fin.size != totalsm*totalch//numbtch :
            print("FILE SIZE ERROR   %s Time:%.2f sec"%(rst_filen, 
                    (time.time() - tstart)))
            exit()
        data_raw = fin.reshape(totalsm, totalch//numbtch, 1).repeat(numbtch, axis=2)            
        if   numbtch == 2 :
            for i in range(numbtch):
                data_raw[:, :, i] >> i*numbits & 0x0f
        elif numbtch == 4 :
            for i in range(numbtch):
                data_raw[:, :, i] >> i*numbits & 0x03
        elif numbtch == 8 :
            for i in range(numbtch):
                data_raw[:, :, i] >> i*numbits & 0x01            
        data_raw = data_raw.reshape(smaple, average, nchan, freqavg).float(
                        ).mean(axis=(1,3))           
    if header['foff'] > 0:  # Reverse the Data if foff > 0
        data_raw = data_raw[:, ::-1]

    #### Cleanning ####
    data_rfi = step_lib_comm.cleanning(data_raw, tthresh, nchan, choff_low ,choff_high, 
                numblock, winsize, totalsm)
    # data_rfi = data_raw.copy()[:, choff_high: nchan-choff_low]

    #### Dedispersion ####
    data_des = np.zeros((smaple, nchan-choff_low-choff_high))
    for i in range(nchan- choff_high - choff_low):
        data_des[:, i] = np.roll(data_rfi.copy()[:, i], int(-delay[i]))

    #### SNR Detecting ####
    med, rms = step_lib_comm.mad(data_des, numblock, winsize)

    #### Smoothing ####
    plot_raw = step_lib_comm.convolve(data_raw, int(plotbc))
    plot_rfi = step_lib_comm.convolve(data_rfi, int(plotbc))
    plot_des = step_lib_comm.convolve(data_des, int(plotbc))

    
    #### Plot PDF File ####
    with PdfPages(rst_filen+'.'+str(header['ibeam'])+'.pdf') as pdf:
        # Calc size of plot #
        smpmax = int(delay[-1]*plotrange)
        if smpmax < 250:
            smpmax = 250
        else:
            smpmax = 500 # smpmax//2*2
        
        for i in range(len(plotime)):
            if maxsm[i] > smaple:
                print("PlotTime =", plotime[i], "exceeded the maximum time of the file", 
                        smaple*header['tsamp']*average)
                continue
            winsel = int(maxsm[i] / winsize)
            if smpmax//2*3 > smaple:
                xlim = 0
                xmax = smaple
                smpmax = smaple//2
            elif maxsm[i] + smpmax//2*3 > smaple:
                xlim = smaple - smpmax*2
                xmax = smaple
            elif maxsm[i] - smpmax//2  < 0:
                xlim = 0
                xmax = smpmax*2
            else:
                xlim = maxsm[i] - smpmax//2
                xmax = maxsm[i] + smpmax//2*3
                
            if maxbc == 1:
                maxsigma = (plot_des.copy().mean(axis=1)[int(maxsm[i])] - med[winsel]*maxbc)/rms[winsel]
            else:
                maxsigma = (plot_des.copy().mean(axis=1)[int(maxsm[i])] - med[winsel]*maxbc)/(
                            rms[winsel]*np.sqrt(maxbc))
            # print()
            # Plot Raw Dedispersion #
            splt.plotdmraw(plot_rfi[int(xlim): int(xmax),:], plot_des[int(xlim): int(xmax),:], 
                            maxsm[i], plotDM, rst_filen, average, freqavg, med[winsel], rms[winsel],
                            nchan-choff_low-choff_high, smaple, smpmax, header, totalsm, delay, 
                            maxsigma, maxbc, choff_low, choff_high, pdf, plotpes, ispsrfits)
        # Plot Raw and RRI data #
        splt.plotraw(plot_raw[:, ::-1], plot_des.copy()[:, ::-1], smaple, rst_filen, average, freqavg, 
                nchan, header, totalsm, choff_low, choff_high, pdf, plotpes, ispsrfits, plotDM, plotbc)
    data_des = []

if __name__ == "__main__":
    print("Init...")
    tstart = time.time()
    PlotReady, FILENAME, PLOTFILE = step_lib_comm.readplotini("frbcfg.ini")
    if PlotReady == 1:
        if len(PLOTFILE) != 0:
            frbplot(PLOTFILE, tstart)
        else:
            if len(FILENAME) != 0:
                for FILE in FILENAME: 
                    # print("Starting...", FILE)           
                    frbplot(FILE, tstart)

        print("End %.2f"%(time.time() - tstart))
    else:
        print("PlotReady set 0")
