#! /usr/bin/env python
import os
import sys
import time
import math
import numpy as np
import readfil
import step_lib_comm
import step_lib_plt as splt
from matplotlib.backends.backend_pdf import PdfPages
import torch

def frbplot(filen, ststart):
    #### Read Config File ####
    (_, NSMAX, _, _, _, tthresh, IGNORE, winsize, choff_low, choff_high, plotpes, plotbc,
        plotime, plotrange, plotDM, average, freqavg, useGPU, BlockSize,
        ) = step_lib_comm.readini("frbcfg.ini")
    choff_high = choff_high//freqavg
    choff_low = choff_low//freqavg
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
    # calc para #
    totalch = header['nchans']
    numbits = header['nbits']
    smptm = header['tsamp']*1e6
    smaple = totalsm//average
    numblock = smaple//winsize
    totalsm = numblock*winsize*average
    smaple = numblock*winsize
    nchan = totalch//freqavg
    smptm *= average
    maxbc = plotbc
    print("Start %s Nchan:%d Nbits:%d TotalSample:%d TotalTime:%.2f sec"%(
            rst_filen, totalch, numbits, totalsm, totalsm*header['tsamp']))
    sys.stdout.flush()

    # Init #
    maxsm = np.zeros(len(plotime))
    data_raw = np.zeros((smaple, nchan), dtype=np.float32)
    plot_rfi = np.zeros((smaple, nchan-choff_low-choff_high), dtype=np.float32)
    plot_des = np.zeros((smaple, nchan-choff_low-choff_high), dtype=np.float32)
    med = np.zeros((numblock), dtype=np.float32)
    rms = np.zeros(numblock, dtype=np.float32)

    # calc time to samples #
    for i in range(len(plotime)):
        maxsm[i] = np.round((float(plotime[i])/header['tsamp'] + plotbc//2)/average)

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

    #### Read Filterbank File ####
    if ispsrfits: # PSRFITS #
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
            sys.stdout.flush()
            exit()
    else:               # BITS NUMBER 1/2/4
        numbtch = 8//numbits
        with open(str(filen),'rb') as fn:
            fn.seek(headsize)
            fin = np.fromfile(fn, dtype=np.uint8, count=totalsm*totalch//numbtch)
        if fin.size != totalsm*totalch//numbtch :
            print("FILE SIZE ERROR   %s Time:%.2f sec"%(rst_filen, 
                    (time.time() - tstart)))
            sys.stdout.flush()
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
    print("Read FILE %.2f"%(time.time() - tstart))
    sys.stdout.flush()
    # Calc Block #
    Blockwz = BlockSize*1024*1024//(nchan*4*winsize)
    Blocksm = Blockwz*winsize
    print("Blocksm =", Blocksm, "Totalsm =", totalsm, "WindowSize =", winsize)
    if Blocksm > smaple :
        Blocksm = smaple
        BlockNum = 1
    else:
        BlockNum = math.ceil(smaple/Blocksm)
    print("Blocksm =", Blocksm, BlockNum, (BlockNum*Blocksm-smaple)/winsize, Blocksm*header['tsamp'])    
    sys.stdout.flush()
    for bnum in range(BlockNum):
        if (bnum+1)*Blocksm + int(delay[-1])> smaple:
            block_tlsm = smaple - bnum*Blocksm
            block_sm = smaple - bnum*Blocksm
            block_nb = block_sm//winsize
        else:
            block_tlsm = Blocksm + int(delay[-1])
            block_sm = Blocksm
            block_nb = block_sm//winsize

        if useGPU == True:
            cuda = torch.device("cuda")
            data_rfi = torch.zeros((block_tlsm, nchan-choff_low-choff_high), dtype=float, device=cuda)
            data_des = torch.zeros((block_tlsm, nchan-choff_low-choff_high), dtype=float, device=cuda)
            # Cleanning #
            data_rfi = step_lib_comm.cleanning_gpu(torch.from_numpy(
                    data_raw[bnum*Blocksm: bnum*Blocksm+block_tlsm]).cuda(), 
                    tthresh, nchan, choff_low ,choff_high, block_nb, winsize, smaple)
            # step_lib_comm.printcuda(cuda)
            print("Clean %.2f"%(time.time() - tstart))
            sys.stdout.flush()
            # Dedispersion #
            data_des = torch.zeros((block_tlsm, nchan-choff_low-choff_high), dtype=float, device=cuda)
            print(data_des.shape, data_rfi.shape, delay[-1], block_tlsm)
            sys.stdout.flush()
            for i in range(nchan- choff_high - choff_low):
                data_des[:, i] = torch.roll(data_rfi[:, i], int(-delay[i]))
            # step_lib_comm.printcuda(cuda)
            print("Dedispersion %.2f"%(time.time() - tstart))
            sys.stdout.flush()
            # SNR Detecting #
            med_bl, rms_bl = step_lib_comm.mad_gpu(data_des[: block_sm].detach().clone(), block_nb, winsize)
            med[bnum*Blockwz: bnum*Blockwz+block_nb] = np.array(med_bl.cpu())
            rms[bnum*Blockwz: bnum*Blockwz+block_nb] = np.array(rms_bl.cpu())
            # step_lib_comm.printcuda(cuda)
            print("MAD %.2f"%(time.time() - tstart))
            sys.stdout.flush()
            # Smoothing #
            plot_rfi[bnum*Blocksm: bnum*Blocksm+block_tlsm] = np.array(step_lib_comm.convolve_gpu(data_rfi, int(plotbc)).cpu())
            plot_des[bnum*Blocksm: bnum*Blocksm+block_tlsm] = np.array(step_lib_comm.convolve_gpu(data_des, int(plotbc)).cpu())
            # step_lib_comm.printcuda(cuda)
            print("Smoothing %.2f"%(time.time() - tstart))
            sys.stdout.flush()    
        else:
            # Cleanning #
            data_rfi = step_lib_comm.cleanning(data_raw, tthresh, nchan, choff_low ,choff_high, 
                        numblock, winsize, totalsm)
            # data_rfi = data_raw.copy()[:, choff_high: nchan-choff_low]
            print("Clean %.2f"%(time.time() - tstart))
            sys.stdout.flush()
            #Dedispersion #
            data_des = np.zeros((smaple, nchan-choff_low-choff_high))
            for i in range(nchan- choff_high - choff_low):
                data_des[:, i] = np.roll(data_rfi.copy()[:, i], int(-delay[i]))
            print("Dedispersion %.2f"%(time.time() - tstart))
            sys.stdout.flush()
            # SNR Detecting #
            med, rms = step_lib_comm.mad(data_des, numblock, winsize)
            print("MAD %.2f"%(time.time() - tstart))
            sys.stdout.flush()
            # Smoothing #
            plot_rfi = step_lib_comm.convolve(data_rfi, int(plotbc))
            plot_des = step_lib_comm.convolve(data_des, int(plotbc))
            print("Smoothing %.2f"%(time.time() - tstart))
            sys.stdout.flush()    
    data_rfi = []
    data_des = []
    #### Plot PDF File ####
    with PdfPages('PLOT'+rst_filen+'.'+str(header['ibeam'])+'.pdf') as pdf:
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
                sys.stdout.flush()
                continue
            winsel = int(maxsm[i] / winsize)
            if smpmax//2*4 > smaple:
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
                
            # if maxbc == 1:
            #     maxsigma = (plot_des.copy().mean(axis=1)[int(maxsm[i])] - med[winsel]*maxbc)/rms[winsel]
            # else:
            #     maxsigma = (plot_des.copy().mean(axis=1)[int(maxsm[i])] - med[winsel]*maxbc)/(
            #                 rms[winsel]*np.sqrt(maxbc))

            # Plot Raw Dedispersion #
            splt.plotdmraw(plot_rfi[int(xlim+int(delay[-1]//2)): int(xmax+int(delay[-1]//2)),:], plot_des[int(xlim): int(xmax),:], 
                            maxsm[i], plotDM, rst_filen, average, freqavg, med[winsel], rms[winsel],
                            nchan-choff_low-choff_high, smaple, smpmax, header, totalsm, delay, 
                            maxbc, choff_low, choff_high, pdf, plotpes, ispsrfits)
        # Plot Raw and RRI data #
        splt.plotraw(data_raw[:, ::-1], plot_des.copy()[:, ::-1], smaple, rst_filen, average, freqavg, 
                nchan, header, totalsm, choff_low, choff_high, pdf, plotpes, ispsrfits, plotDM, plotbc)
    print("Save PDF %.2f"%(time.time() - tstart))
    sys.stdout.flush()   

if __name__ == "__main__":
    print("Init...")
    sys.stdout.flush()
    tstart = time.time()
    PlotReady, FILENAME, PLOTFILE = step_lib_comm.readplotini("frbcfg.ini")
    if PlotReady == 1:
        if len(PLOTFILE) != 0:
            frbplot(PLOTFILE, tstart)
        else:
            if len(FILENAME) != 0:
                for FILE in FILENAME:   
                    frbplot(FILE, tstart)
                    print("Finish %s %.2f"%(FILE, time.time() - tstart))
                    sys.stdout.flush()
        print("End Plot %.2f"%(time.time() - tstart))
    else:
        print("PlotReady 0, No process")
