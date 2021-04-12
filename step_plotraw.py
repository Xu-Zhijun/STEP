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
## Debug ##
# plotimage = False # Plot image in given time
# fakefrb = False # Gen training dataset
# fakenofrb = False # Gen without FRB
# fakerfi = False # Gen RFI

def frbplot(filen, ststart):
    #### Read Config File ####
    (THRESH, NSMAX, _, _, _, tthresh, IGNORE, winsize, choff_low, choff_high, plotpes,
        plotbc, plotime, plotrange, plotDM, average, freqavg, useGPU, BlockSize,
        ) = step_lib_comm.readini("frbcfg.ini")
    choff_high = choff_high//freqavg
    choff_low = choff_low//freqavg

    #### READ HEADER ####
    _, rst_filen = os.path.split(filen)
    ispsrfits = False
    headsize = 0
    MAXSNR = 0
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
    sample = totalsm//average
    numblock = sample//winsize
    totalsm = numblock*winsize*average
    sample = numblock*winsize
    nchan = totalch//freqavg
    smptm *= average
    maxbc = plotbc
    # Init #
    maxsm = np.zeros(len(plotime))
    data_raw = np.zeros((sample, nchan), dtype=np.float32)
    plot_rfi = np.zeros((sample, nchan-choff_low-choff_high), dtype=np.float32)
    plot_des = np.zeros((sample, nchan-choff_low-choff_high), dtype=np.float32)
    med = np.zeros((numblock, 1), dtype=np.float32)
    rms = np.zeros(numblock, dtype=np.float32)
    data_psr = np.zeros((sample, nchan), dtype=np.float32)
    frbsignal = np.zeros((3, nchan), dtype=np.float32)

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
    delayint = 4148.741601*(chfrq**(-2) - (higch)**(-2))/smptm
    # print(plotDM, lowch, higch, smptm, 4148.741601*323.2*(0.200**(-2) - (0.2008)**(-2)))
    # exit()
    if plotDM > 0:
        delayMax = math.ceil(delay[-1]/winsize)*winsize
    else:
        delayMax = 0
    print("Start %s Nchan:%d Nbits:%d TotalSample:%d TotalTime:%.2f sec, Maxdelay:%d, DelayMax:%d"%(
            rst_filen, totalch, numbits, totalsm, totalsm*header['tsamp'], delay[-1], delayMax))
    sys.stdout.flush()

    # Read PSRFITS #
    if ispsrfits: # PSRFITS #
        data_psr = psrdata[:totalsm, :nchan*freqavg].reshape(sample, average, nchan, freqavg).mean(axis=(1,3))

    # Calc Block Number #
    Blockwz = BlockSize*1024*1024//(nchan*4*winsize)
    Blocksm = Blockwz*winsize
    # print("Blocksm =", Blocksm, "Totalsm =", totalsm, "WindowSize =", winsize)
    if Blocksm > sample :
        Blocksm = sample
        BlockNum = 1
    else:
        BlockNum = math.ceil(sample/Blocksm)
    if Blocksm < delayMax:
        print("Warning!!! Max delay is larger than Block Size!!!!")
        sys.stdout.flush()
    # print("Blocksm =", Blocksm, BlockNum, (BlockNum*Blocksm-sample)/winsize, Blocksm*header['tsamp'])    
    sys.stdout.flush()
    #### Main loop ####    
    with PdfPages('PLOT'+rst_filen+'.'+str(header['ibeam'])+'.pdf') as pdf:
        for bnum in range(BlockNum):
            # calc current blocksize #
            if (bnum+1)*Blocksm + delayMax> sample:
                block_tlsm = sample - bnum*Blocksm
                # block_sm = sample - bnum*Blocksm
                # block_nb = block_sm//winsize
            else:
                block_tlsm = Blocksm + delayMax

            if (bnum+1)*Blocksm > sample:
                block_sm = sample - bnum*Blocksm
            else:
                block_sm = Blocksm
            block_nb = block_sm//winsize
            if numbits == 8:
                header_offset = headsize+bnum*Blocksm*average*totalch
            elif numbits == 16:
                header_offset = headsize+bnum*Blocksm*average*totalch*2
            elif numbits == 32:
                header_offset = headsize+bnum*Blocksm*average*totalch*4
            elif numbits == 4:
                header_offset = headsize+bnum*Blocksm*average*totalch//2
            elif numbits == 2:
                header_offset = headsize+bnum*Blocksm*average*totalch//4
            elif numbits == 1:
                header_offset = headsize+bnum*Blocksm*average*totalch//8
            
            # read file #
            # print(block_tlsm, block_sm, block_nb, header_offset)
            if ispsrfits: # PSRFITS
                data_raw = data_psr[bnum*Blocksm: bnum*Blocksm+block_tlsm]
                # data_raw = psrdata[:totalsm, :nchan*freqavg].reshape(sample, average, nchan, freqavg).mean(axis=(1,3))
            else:
                data_raw = step_lib_comm.read_file(filen, data_raw, numbits, header_offset, 
                                block_tlsm*average*totalch, block_tlsm, average, nchan, freqavg, tstart)
            if header['foff'] > 0:  # Reverse the Data if foff > 0
                data_raw = data_raw[:, ::-1]
            print("%d/%d Read FILE %.2f"%(bnum+1, BlockNum, time.time() - tstart))
            sys.stdout.flush()
            # GPU or CPU #
            if useGPU == True:
                # init tensor #
                cuda = torch.device("cuda")
                data_rfi = torch.zeros((block_tlsm, nchan-choff_low-choff_high), dtype=float, device=cuda)
                data_des = torch.zeros((block_tlsm, nchan-choff_low-choff_high), dtype=float, device=cuda)

                # Cleanning #
                data_tmp = step_lib_comm.cleanning(data_raw, tthresh, nchan, choff_low ,choff_high, 
                            block_tlsm//winsize, winsize, block_tlsm, IGNORE, plotbc)
                data_rfi = torch.from_numpy(data_tmp).cuda()
                # step_lib_comm.printcuda(cuda)
                print("%d/%d Clean %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                sys.stdout.flush()

                # Dedispersion #
                sys.stdout.flush()
                for i in range(nchan- choff_high - choff_low):
                    data_des[:, i] = torch.roll(data_rfi[:, i], int(-delay[i]))
                print("%d/%d Dedispersion %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                sys.stdout.flush()

                # SNR Detecting #
                med_bl, rms_bl = step_lib_comm.mad_gpu(
                                        data_des[: block_sm].detach().clone().mean(dim= 1).view(block_nb, winsize), 
                                        block_nb, winsize)
                med[bnum*Blockwz: bnum*Blockwz+block_nb, :] = np.array(med_bl.cpu())
                rms[bnum*Blockwz: bnum*Blockwz+block_nb] = np.array(rms_bl.cpu())
                print("%d/%d MAD %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                sys.stdout.flush()

                # Smoothing #
                plot_rfi = data_raw[:, choff_high: nchan-choff_low] #np.array(data_rfi.cpu())
                plot_des = np.array(data_des.cpu())
                # if bnum == 0 and BlockNum != 1:
                #     plot_rfi[: block_tlsm] = np.array(
                #                             step_lib_comm.convolve_gpu(data_rfi, int(plotbc)).cpu())
                #     plot_des[: block_tlsm] = np.array(
                #                             step_lib_comm.convolve_gpu(data_des, int(plotbc)).cpu())                    
                # else:
                #     plot_rfi[bnum*Blocksm: bnum*Blocksm+block_sm] = np.array(
                #                             step_lib_comm.convolve_gpu(data_rfi[: block_sm], int(plotbc)).cpu())
                #     plot_des[bnum*Blocksm: bnum*Blocksm+block_sm] = np.array(
                #                             step_lib_comm.convolve_gpu(data_des[: block_sm], int(plotbc)).cpu())
                # print("%d/%d Smoothing %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                # sys.stdout.flush()    
            else:
                # init array #
                data_rfi = np.zeros((block_tlsm, nchan-choff_low-choff_high), dtype=np.float32)
                data_des = np.zeros((block_tlsm, nchan-choff_low-choff_high), dtype=np.float32)

                # Cleanning #
                data_rfi = step_lib_comm.cleanning(data_raw, tthresh, nchan, choff_low ,choff_high, 
                            block_tlsm//winsize, winsize, block_tlsm, IGNORE, plotbc)
                print("%d/%d Clean %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                sys.stdout.flush()

                #Dedispersion #
                for i in range(nchan- choff_high - choff_low):
                    data_des[:, i] = np.roll(data_rfi[:, i], int(-delay[i]))
                print("%d/%d Dedispersion %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                sys.stdout.flush()

                # SNR Detecting #
                med_bl, rms_bl = step_lib_comm.mad(data_des[: block_sm].copy().mean(axis= 1).reshape(block_nb, winsize),
                                                block_nb, winsize)
                med[bnum*Blockwz: bnum*Blockwz+block_nb, :] = med_bl
                rms[bnum*Blockwz: bnum*Blockwz+block_nb] = rms_bl
                print("%d/%d MAD %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                sys.stdout.flush()

                # Smoothing #
                plot_rfi = data_rfi
                plot_des = data_des
                # if bnum == 0 and BlockNum != 1:
                #     plot_rfi[: block_tlsm] = step_lib_comm.convolve(data_rfi, int(plotbc))
                #     plot_des[: block_tlsm] = step_lib_comm.convolve(data_des, int(plotbc))
                # else:
                #     plot_rfi[bnum*Blocksm: bnum*Blocksm+block_sm] = step_lib_comm.convolve(data_rfi[: block_sm], int(plotbc))
                #     plot_des[bnum*Blocksm: bnum*Blocksm+block_sm] = step_lib_comm.convolve(data_des[: block_sm], int(plotbc))
                # print("%d/%d Smoothing %.2f"%(bnum+1, BlockNum, time.time() - tstart))
                # sys.stdout.flush() 

            # sub plot #
            # if BlockNum != 1:                
            #     if bnum == 0:
            #         # print("First Block", Blocksm, block_tlsm)
            #         splt.plotraw((plot_rfi[Blocksm: block_tlsm])[:, ::-1], 
            #                     (plot_des[Blocksm: block_tlsm])[:, ::-1], 
            #                     (block_tlsm-Blocksm), rst_filen, average, freqavg, nchan, header,  
            #                     (block_tlsm-Blocksm)*average, choff_low, choff_high, pdf, plotpes, 
            #                     ispsrfits, plotDM, plotbc, 0, winsize) 
            if len(plotime) == 0:
                for nb in range(block_nb):
                    plot_offset = nb*winsize #+ winsize//2 #+ bnum*Blocksm
                    # if plot_offset+winsize > sample:
                    #     print('plot_offset+winsize > sample')
                    #     continue
                    sigma = (plot_des[plot_offset: plot_offset + winsize].copy().mean(axis=1) -
                            med[bnum*Blockwz + nb])/rms[bnum*Blockwz + nb]
                    # sigma = (plot_des[plot_offset: plot_offset + winsize].copy().mean(axis=1) -
                    #         med[bnum*Blockwz + nb]*maxbc)/(rms[bnum*Blockwz + nb]*np.sqrt(maxbc))
                    # print(sigma.shape, sigma)
                    # if fakefrb:
                    #     splt.fakefrb((data_raw[plot_offset: plot_offset + winsize]),
                    #     winsize, delayint, fakenofrb, fakerfi, nchan,'%s.%d.%d.%d'%(rst_filen, 
                    #     header['ibeam'], bnum, nb))

                    maxsigma = np.max(sigma)
                    if MAXSNR < maxsigma:
                        MAXSNR = maxsigma
                    if maxsigma > THRESH :
                        print("maxsigma =", maxsigma, "Block =", bnum,
                            "window =", nb)
                        sys.stdout.flush()
                        
                        # with PdfPages('PLOT'+rst_filen+'.'+str(header['ibeam'])+'.pdf') as pdf:
                        # pdf = []
                        splt.plotraw((plot_rfi[plot_offset: plot_offset + winsize])[:, ::-1], 
                                (plot_des[plot_offset: plot_offset + winsize])[:, ::-1], 
                                winsize, rst_filen, average, freqavg, nchan, header, winsize*average, 
                                choff_low, choff_high, pdf, plotpes, ispsrfits, plotDM, plotbc, 
                                plot_offset*header['tsamp']*average, winsize, maxsigma)  

        #### Plot PDF File ####
        if len(plotime) != 0:
        # with PdfPages('PLOT'+rst_filen+'.'+str(header['ibeam'])+'.pdf') as pdf:
            # Calc size of plot #
            smpmax = 336 #int(delay[-1]*plotrange)
            if smpmax < 250:
                smpmax = 250
            else:
                smpmax = 500 # smpmax//2*2
            
            for i in range(len(plotime)):
                if maxsm[i] > sample:
                    print("PlotTime =", plotime[i], "exceeded the maximum time of the file", 
                            sample*header['tsamp']*average)
                    sys.stdout.flush()
                    continue
                winsel = int(maxsm[i] / winsize)
                if smpmax//2*4 > sample:
                    xlim = 0
                    xmax = sample
                    smpmax = sample//2
                elif maxsm[i] + smpmax//2*3 > sample:
                    xlim = sample - smpmax*2
                    xmax = sample
                elif maxsm[i] - smpmax//2  < 0:
                    xlim = 0
                    xmax = smpmax*2
                else:
                    xlim = maxsm[i] - smpmax//2
                    xmax = maxsm[i] + smpmax//2*3
                # save frb #
                for ts in range(336):
                    frboffset = int(maxsm[i]) + int(delay[ts])
                    frbsignal[:, ts] = plot_rfi[frboffset-2:frboffset+1, ts]
                np.save('frb.%s.%d'%(rst_filen, header['ibeam']), frbsignal)
                
                # Plot Raw Dedispersion #
                # if plotimage:
                #     splt.plotpng(plot_rfi[int(xlim+int(delay[-1]//2)): int(xmax+int(delay[-1]//2)), :],
                #             plot_des[int(xlim): int(xmax), :], smpmax, rst_filen+'.'+str(header['ibeam']))
                # else:
                splt.plotdmraw(plot_rfi[int(xlim+int(delay[-1]//2)): int(xmax+int(delay[-1]//2)),:], 
                                plot_des[int(xlim): int(xmax),:], 
                                maxsm[i], plotDM, rst_filen, average, freqavg, med[winsel], rms[winsel],
                                nchan-choff_low-choff_high, sample, smpmax, header, totalsm, delay, 
                                maxbc, choff_low, choff_high, pdf, plotpes, ispsrfits)
        # Plot Raw and RFI data #
        # splt.plotraw(plot_rfi[:, ::-1], plot_des[:, ::-1], sample, rst_filen, average, freqavg, 
        #         nchan, header, totalsm, choff_low, choff_high, pdf, plotpes, ispsrfits, plotDM, plotbc,
        #         0, winsize, MAXSNR)
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
                FILENAME.sort()
                for FILE in FILENAME:   
                    frbplot(FILE, tstart)
                    print("Finish %s %.2f"%(FILE, time.time() - tstart))
                    sys.stdout.flush()
        print("End Plot %.2f"%(time.time() - tstart))
    else:
        print("PlotReady 0, No process")
