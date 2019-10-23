#! /usr/bin/env python
import os
import sys
import time
import readfil
import astropy.io.fits as pyfits
import numpy as np

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
    if (FREQAVG == 0 or AVERAGE == 0) :
        print("AVERAGE or FREQAVG can't be Zero !!!")
        exit()
    return (THRESH, NSMAX, LODM, HIDM, DDM, RFITHR, IGNORE, WINDOWSIZE, CHOFF_LOW, 
            CHOFF_HIGH, PlotPersent, PlotBoxcar, PlotTime, Plotrange, PlotDM, AVERAGE, FREQAVG)

def convolve(dn, boxcar):
    conv = dn.copy()
    for i in range(1, boxcar):
        # conv[i:] += dn[:-i]
        conv += np.roll(dn, i, axis= 0)
    return conv

def mad(din, nbl, wsize):
    tmp_des = np.sort(din.copy().mean(axis= 1).reshape(nbl, wsize), axis=1)
    med = tmp_des[:, wsize//2].reshape(nbl, 1)
    rms = np.sort(np.abs(tmp_des - med))[:, wsize//2] #1.4826*
    return med, rms

def cleanning(din, tthresh, totalch, choff_low, choff_high, nbl, wsize, sample):
    #### Remove RFI in time ####
    nch = totalch-choff_low-choff_high
    data_rfi = din.copy()[:, choff_high: totalch-choff_low]
    med, rms = mad(data_rfi, nbl, wsize)
    sigma = ((data_rfi.copy().mean(axis = 1).reshape(nbl, wsize) - med
                )/rms.reshape(nbl, 1)).reshape(-1)
    # data_rfi[np.where(sigma > tthresh)] = np.random.chisquare(wsize, 
    #                     nch)/wsize*np.sqrt((med**2).mean())
    data_rfi[np.where(sigma > tthresh)] =  data_rfi.copy().mean(axis=0)

    #### Remove RFI in frequency ####
    tmp_frq = np.sort(data_rfi.copy().mean(axis= 0), axis=0)  
    med_frq = tmp_frq[nch//2]
    rms_frq = np.sort(np.abs(tmp_frq - med_frq))[nch//2]
    sigma_frq = ((data_rfi.copy().mean(axis = 0) - med_frq)/rms_frq).reshape(-1)
    # data_rfi.transpose()[np.where(sigma_frq > tthresh)] = np.random.chisquare(nch, 
    #                     sample)/nch*np.sqrt((med_frq**2).mean())
    data_rfi.transpose()[np.where(sigma_frq > tthresh)] = data_rfi.copy().mean(axis=1)
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
        # print(header, psrdata.shape, nsubints, nsampsub, shp, polnorder, data.shape)
    # print(data.shape, psrdata.shape)
    return header, data[:, ::-1]