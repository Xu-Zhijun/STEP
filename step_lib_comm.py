#! /usr/bin/env python
import os
import sys
import time
import readfil
import astropy.io.fits as pyfits
import numpy as np
import torch

def printcuda(cuda):
    print("GPU Memory Using: ",
    torch.cuda.memory_allocated(cuda)//(1024*1024), torch.cuda.max_memory_allocated(cuda)//(1024*1024), 
    torch.cuda.memory_cached(cuda)//(1024*1024), torch.cuda.max_memory_cached(cuda)//(1024*1024))

def readplotini(inifile):
    FILENAME = []
    FITSFILE = []
    PlotReady = 0
    PLOTFILE = []
    with open(inifile,'r') as fd:
        all_lines = fd.readlines()
    for i in range(len(all_lines)):
        #### Skip Empty Line ####
        if len(all_lines[i].split()) == 0:
            continue
        #### Skip # Line ####
        elif "#" in all_lines[i].split()[0]:
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
            elif fil.endswith(".fits"):
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
    PlotPersent = 1.0
    useGPU = True #False #
    BlockSize = 1000
    with open(inifile,'r') as fd:
        all_lines = fd.readlines()
    for i in range(len(all_lines)):
        #### Skip Empty Line ####
        if len(all_lines[i].split()) == 0:
            continue
        #### Skip # Line ####
        elif "#" in all_lines[i].split()[0]:
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
            # IGNORE = int(all_lines[i].split()[2])
            for s in range(len(all_lines[i].split()) - 2):
                IGNORE.append(all_lines[i].split()[2+s]) 
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
        elif 'PlotPersent' in all_lines[i]:
            PlotPersent = float(all_lines[i].split()[2]) 
            if PlotPersent <= 0:
                print("PlotPersent can't <= 0")
                exit()
            elif PlotPersent > 1:
                print("PlotPersent can't > 1")
                exit()
        elif 'PlotBoxcar' in all_lines[i]:
            PlotBoxcar = float(all_lines[i].split()[2])     
        elif 'BlockSize' in all_lines[i]:
            BlockSize = int(all_lines[i].split()[2])
        elif 'useGPU' in all_lines[i]:
            if int(all_lines[i].split()[2]) == 0:
                print("Using CPU")
                useGPU = False
            else:
                print("Using GPU")
                useGPU = True
            sys.stdout.flush()
                
    if (FREQAVG == 0 or AVERAGE == 0) :
        print("AVERAGE or FREQAVG can't be Zero !!!")
        exit()
    return (THRESH, NSMAX, LODM, HIDM, DDM, RFITHR, IGNORE, WINDOWSIZE, CHOFF_LOW, 
            CHOFF_HIGH, PlotPersent, PlotBoxcar, PlotTime, Plotrange, PlotDM, AVERAGE, 
            FREQAVG, useGPU, BlockSize,)

def convolve(dn, boxcar):
    conv = dn.copy()
    for i in range(1, boxcar):
        # conv[i:] += dn[:-i]
        # conv[:i] += dn[-i:]
        conv += np.roll(dn, i, axis = 0)
    return conv

def convolve_gpu(dn, boxcar):
    conv = dn.detach().clone()
    for i in range(1, boxcar):
        # conv[i:] += dn[:-i]
        # conv[:i] += dn[-i:]
        conv += torch.roll(dn, i, dims = 0)
    return conv

def mad(din, nbl, wsize):
    # tmp_des = np.sort(din.copy().mean(axis= 1).reshape(nbl, wsize), axis=1)
    # med = tmp_des[:, wsize//2].reshape(nbl, 1)
    # rms = np.sort(np.abs(tmp_des - med))[:, wsize//2] #
    # din = din.mean(axis= 1).reshape(nbl, wsize)
    med = np.median(din, axis=1).reshape(nbl, 1)
    rms = np.median(np.abs(din-med), axis=1)
    return med, 1.4826*rms

def mad_gpu(din, nbl, wsize):
    # tmp_des, _ = torch.sort(din.mean(dim= 1).view(nbl, wsize), dim=1)
    # med = tmp_des[:, wsize//2].view(nbl, 1)
    # tmp_des, _ = torch.sort(torch.abs(tmp_des - med))
    # rms = tmp_des[:, wsize//2] #
    # din = din.mean(dim= 1).view(nbl, wsize)
    med, _ = torch.median(din, 1)
    med = med.view(nbl, 1)
    rms, _ = torch.median(torch.abs(din-med), 1)
    return med, 1.4826*rms

def cleanning(din, tthresh, totalch, choff_low, choff_high, nbl, wsize, sample, ignore, plotbc):
    #### Remove offset channel ####
    nch = totalch-choff_low-choff_high
    data_conv = din.copy()[:, choff_high: totalch-choff_low]
    #### Convolve ####
    data_rfi = convolve(data_conv, int(plotbc))
    #### Ignore channels ####
    channel_med = np.median(data_rfi, axis=1)
    for i in range(len(ignore)):
        for s in range(5):
            data_rfi.transpose()[int(ignore[i])-2+s] = (
                np.random.normal(channel_med.mean(), np.std(channel_med), data_rfi.shape[0]))
            #(med_rfi.reshape(1, -1)).repeat(5, axis=0)
    #### Remove RFI in time ####
    # med_rfi = np.median(data_rfi.copy(), axis=1)
    med_tim = np.median(data_rfi.copy(), axis=0)
    # med, rms = mad(data_rfi, nbl, wsize)
    # sigma = ((data_rfi.copy().mean(axis = 1).reshape(nbl, wsize) - med
    #             )/rms.reshape(nbl, 1)).reshape(-1)
    # # data_rfi[np.where(sigma > tthresh)] = np.random.chisquare(wsize, 
    # #                     nch)/wsize*np.sqrt((med**2).mean())
    # data_rfi[np.where(sigma > tthresh)] =  data_rfi.copy().mean(axis=0)
    data_time = data_rfi.copy().mean(axis= 1).reshape(nbl, wsize)
    med_time, rms_time = mad(data_time, nbl, wsize)
    sigma_time = ((data_time - med_time)/rms_time.reshape(nbl, 1)).reshape(-1)
    data_rfi[np.where(sigma_time > tthresh)] = med_tim

    #### Remove RFI in frequency ####
    # data_frq = data_rfi.copy().mean(axis= 0)
    # med_frq = np.median(data_frq)
    # rms_frq = np.median(np.abs(data_frq - med_frq))
    # sigma_frq = ((data_frq - med_frq)/rms_frq).reshape(-1)
    # data_rfi.transpose()[np.where(sigma_frq > tthresh)] = med_rfi
    # print(med_rfi.shape, data_rfi.transpose().shape, ignore)
    # data_rfi.transpose()[ignore] = med_rfi
    return data_rfi

def cleanning_gpu(din, tthresh, totalch, choff_low, choff_high, nbl, wsize, sample):
    #### Remove RFI in time ####
    nch = totalch-choff_low-choff_high
    data_rfi = din[:, choff_high: totalch-choff_low]
    # data_rfi = data_rfi - data_rfi.mean(dim = 0)
    # med, rms = mad_gpu(data_rfi.detach().clone(), nbl, wsize)
    # sigma = ((data_rfi.mean(dim = 1).view(nbl, wsize) - med
    #             )/rms.view(nbl, 1)).view(-1)
    # data_rfi[torch.where(sigma > tthresh)] =  data_rfi.mean(dim=0)

    # #### Remove RFI in frequency ####
    # tmp_frq, _ = torch.sort(data_rfi.mean(dim= 0), dim=0)  
    # med_frq = tmp_frq[nch//2]
    # tmp_frq, _ = torch.sort(torch.abs(tmp_frq - med_frq))
    # rms_frq = tmp_frq[nch//2]
    # sigma_frq = ((data_rfi.mean(dim = 0) - med_frq)/rms_frq).view(-1)
    # data_rfi.transpose()[torch.where(sigma_frq > tthresh)] = data_rfi.mean(dim=1)
    return data_rfi

# def disbar(max, dn):
#     jd = '\r %2d%% [%s%s]'
#     a = '*'* np.ceil(dn*100/max)
#     b = ' '* ((max-dn)*100//max)
#     c = (dn/max)*100+1
#     print(jd % (c,a,b), end="", flush=True)

def read_psrfits(psrfits_file, ststart):
    """
    Modified from presto prsfit.py
    """
    header = {'ibeam':0, 'nbeams':1,}
    print("Reading...", psrfits_file, time.time() - ststart)
    sys.stdout.flush()
    with open (psrfits_file,'rb') as fn:
        psr01 = pyfits.open(fn, mode='readonly', memmap=True)
        fits_header = psr01['PRIMARY'].header
        sub_header = psr01['SUBINT'].header
        header['telescope_id'] = fits_header['TELESCOP']
        header['machine_id'] = fits_header['BACKEND']
        header['source_name'] = fits_header['SRC_NAME']
        header['src_raj'] = float(fits_header['RA'].replace(':',''))
        header['src_dej'] = float(fits_header['DEC'].replace(':',''))
        header['tstart'] = (fits_header['STT_IMJD'] + fits_header['STT_SMJD']/86400.0 + 
                            fits_header['STT_OFFS']/86400.0)
        header['fch1'] = (fits_header['OBSFREQ'] + np.abs(fits_header['OBSBW'])/2.0 - 
                            np.abs(sub_header['CHAN_BW'])/2.0)
        header['foff'] = -1.0*np.abs(sub_header['CHAN_BW'])
        header['nchans'] = sub_header['NCHAN']
        header['nbits'] =  sub_header['NBITS']
        header['tsamp'] = sub_header['TBIN']
        header['nifs'] = sub_header['NPOL']
        header['totalsm'] = sub_header['NSBLK']*sub_header['NAXIS2']
        nsampsub = sub_header['NSBLK']
        nsubints = sub_header['NAXIS2'] 
        numpolns = sub_header['NPOL']
        polnorder = sub_header['POL_TYPE']
        data = np.zeros((header['totalsm'], header['nchans']), dtype=np.float32)
        for i in range(nsubints):
            psrdata = psr01['SUBINT'].data[i]['DATA']
            shp = psrdata.squeeze().shape
            if (len(shp)==3 and shp[1]==numpolns and polnorder == 'IQUV'):
                # print("Polarization is IQUV, just using Stokes I")
                data[i*nsampsub: (i+1)*nsampsub]= psrdata[:,0,:].squeeze()
            else:
                data[i*nsampsub: (i+1)*nsampsub] = np.asarray(psrdata.squeeze())
    return header, data[:, ::-1]

def read_file(filen, data_raw, numbits, headsize, countsize, smaple, average, 
            nchan, freqavg, tstart):
    if numbits >= 8:    # BITS NUMBER 8/16/32
        with open(str(filen),'rb') as fn:
            fn.seek(headsize)
            if   numbits == 32:
                data_raw = np.fromfile(fn, dtype=np.float32, count=countsize)
            elif numbits == 16:
                data_raw = np.fromfile(fn, dtype=np.uint16, count=countsize)
            elif numbits == 8:
                data_raw = np.fromfile(fn, dtype=np.uint8, count=countsize)
        if data_raw.size != countsize:
            print("FILE SIZE ERROR %d / %d  %s Time:%.2f sec"%(data_raw.size, 
                    countsize, filen, (time.time() - tstart)))
            sys.stdout.flush()
            exit()
        data_raw = data_raw.reshape(smaple, average, nchan, freqavg).mean(axis=(1,3))
    else:               # BITS NUMBER 1/2/4
        numbtch = 8//numbits
        with open(str(filen),'rb') as fn:
            fn.seek(headsize)
            data_raw = np.fromfile(fn, dtype=np.uint8, count=countsize//numbtch)
        if data_raw.size != countsize//numbtch :
            print("FILE SIZE ERROR   %s Time:%.2f sec"%(filen, 
                    (time.time() - tstart)))
            sys.stdout.flush()
            exit()
        data_raw = data_raw.reshape(smaple*average, (nchan*freqavg)//numbtch, 1).repeat(numbtch, axis=2)            
        if   numbtch == 2 :
            for i in range(numbtch):
                data_raw[:, :, i] >> i*numbits & 0x0f
        elif numbtch == 4 :
            for i in range(numbtch):
                data_raw[:, :, i] >> i*numbits & 0x03
        elif numbtch == 8 :
            for i in range(numbtch):
                data_raw[:, :, i] >> i*numbits & 0x01            
        data_raw = data_raw.reshape(smaple, average, nchan, freqavg).mean(axis=(1,3))
    return data_raw           