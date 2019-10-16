#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import readfil

def plotwinx(axes, totalch, fn):
    ax = axes.twinx()
    ax.set_ylabel("Channel", color='w')
    ax.set_ylim(totalch, 0)
    ax.tick_params(colors='w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['bottom'].set_color('w')
    ax.plot((fn.mean(axis=0)), np.arange(totalch-1, -0.1, -1), color='w', linewidth=1)
    return ax

def plotdmraw(finn, dess, pltime, pldm, filname, avg, freqavg, med, rms, totalch, smap, 
            smpmax, header, totalsm, delay, sigma, maxbc, choff_low, choff_high, pdf):
    viewdw = smpmax//4*2 
    #### Read Para ####
    if header['foff'] < 0:
        ymax = header['fch1'] + header['foff']*choff_high*freqavg
        ymin = header['fch1'] + header['foff']*header['nchans'] - header['foff']*choff_low*freqavg
    else:
        ymax = header['fch1'] + header['foff']*header['nchans'] - header['foff']*choff_high*freqavg
        ymin = header['fch1'] + header['foff']*choff_low*freqavg   
    smpt = header['tsamp']*1e6
    freqctr = (ymax+ymin)/2
    timeoff = pltime*header['tsamp']*avg
    # print(timeoff, pltime)
    if header['telescope_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 64, 65]:
        telescope_id = readfil.ids_to_telescope[header['telescope_id']]
    else:
        telescope_id = 'Unknow'
    if header['machine_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 20, 64, 65]:
        machine_id = readfil.ids_to_machine[header['machine_id']]
    else:
        machine_id = 'Unknow'
    #### Clac MaxSNR ####
    snr = (dess[:smpmax].copy().mean(axis=1)-med*maxbc)/(rms*np.sqrt(maxbc))
    maxsigma = np.max(snr)
    sampleoff = (np.argmax(snr) - smpmax//2)*smpt*1e-6*avg
    # print(smpmax, np.argmax(snr))
    #### SET DATA ####
    fn = (dess[int(smpmax//2-viewdw): int(smpmax//2+viewdw)])[:, ::-1]
    fn2 = (finn[int(smpmax//2-viewdw): int(smpmax//2+viewdw)])[:, ::-1]
    xlim = float(pltime-smpmax//4*2)*smpt*1e-6*avg
    xmax = float(pltime+smpmax//4*2)*smpt*1e-6*avg
    xstick = [xlim+(xmax-xlim)/4, xlim+(xmax-xlim)/2, xlim+(xmax-xlim)*3/4]
    #### Plot ####
    fig, axes = plt.subplots(3, 2, gridspec_kw={'height_ratios':[1,3,3], 'width_ratios':[10, 1]})
    fig.set_facecolor('k')
    fig.set_size_inches(15, 10)
    plt.subplots_adjust(wspace= 0.01, hspace= 0.02, left=0.06, bottom=0.07, right=0.95, top=0.88)
    #### Plot TEXT ####
    fig.suptitle("SHAO_FRB results for:  %s Offset: %8.4fs %s DM: %4.4f \
    Average:%3d %sFreqAvg:%3d  %sBOXCAR:%3d %sBeam: %3d/%3d\n\n\n\n\n"%(filname.ljust(25,' '),  
        timeoff, ' '.ljust(3,' '), pldm, avg, ' '.ljust(3,' '), freqavg, ' '.ljust(3,' '), maxbc,
        ' '.ljust(3,' '), header['ibeam'], header['nbeams']), color='1') 
    fig.text(0.2, 0.94, "Source: %s"%header['source_name'], color='1', size=12)
    fig.text(0.2, 0.92, "Telscope: %s"%telescope_id, color='1', size=12)
    fig.text(0.2, 0.90, "Instrument: %s"%machine_id, color='1', size=12)
    fig.text(0.4, 0.94, "RA (J2000): %s"%str(header['src_raj']), color='1', size=12)
    fig.text(0.4, 0.92, "DEC (J2000): %s"%str(header['src_dej']), color='1', size=12)
    fig.text(0.4, 0.90, "MJD(bary): %s"%str(header['tstart']), color='1', size=12)    
    fig.text(0.65, 0.94, "N samples: %s"%str(totalsm), color='1', size=12)
    fig.text(0.65, 0.92, "Sampling time: %s us"%str(smpt), color='1', size=12)
    fig.text(0.65, 0.90, "Freq(ctr): %s MHz"%str(freqctr), color='1', size=12)
    ### SET STYLE ####
    for yl in range(3):
        axes[yl, 1].set_yticks([])    
        for xl in range(2):
            if yl == 2 or xl != 1 :
                axes[yl, xl].set_facecolor('k')
                axes[yl, xl].tick_params(colors='w')
                axes[yl, xl].spines['right'].set_color('w')
                axes[yl, xl].spines['left'].set_color('w')    
                axes[yl, xl].spines['top'].set_color('w')
                axes[yl, xl].spines['bottom'].set_color('w')
    axes[0,1].set_facecolor('k')
    axes[1,1].set_facecolor('k')
    #### Plot Flux ####
    axes[0,0].set_ylabel("Flux", color='w')
    fig.text(0.875, 0.85, 'SNR: %4.3f'%sigma, color='1')
    fig.text(0.875, 0.82, 'MAXSNR: %4.3f'%maxsigma, color='1')
    fig.text(0.875, 0.79, 'OFFSET: %fs'%sampleoff, color='1')
    axes[0,0].plot(np.arange(-viewdw, viewdw), (fn.mean(axis=1)), color='w', linewidth=1)
    axes[0,0].set_xlim(-viewdw, viewdw)
    axes[0,0].set_xticks([-viewdw/2, 0, viewdw/2])
    #### Plot Result ####
    re1 = axes[1,0].imshow(np.transpose(fn), aspect = 'auto', origin = 'lower',
            extent = [-viewdw, viewdw, ymin, ymax],
            cmap = 'plasma')  # viridis, magma, Blues
    axes[1,0].set_ylabel("Frequency (MHz)", color='w')
    axes[1,0].tick_params(colors='w')
    axes[1,0].set_xticks([-viewdw/2, 0, viewdw/2]) 
    #### Plot Raw data ####
    re2 = axes[2,0].imshow(np.transpose(fn2), aspect = 'auto', origin = 'lower',
            extent = [xlim, xmax, ymin, ymax],
            cmap = 'Blues') # viridis, magma, Blues
    axes[2,0].set_ylabel("Frequency (MHz)", color='w')
    axes[2,0].set_xlabel("Time (s)", color='w')
    axes[2,0].tick_params(colors='w')
    axes[2,0].set_xticks(xstick) 
    #### Plot Raw Flux ####
    plotwinx(axes[2,1], totalch, fn2)
    axes[2,1].set_xlabel("Flux", color='w')
    #### Plot colorbar ####
    pst1 = fig.add_axes([0.875, 0.425, 0.012, 0.33])
    pst1.set_facecolor('k')
    cb1 = fig.colorbar(re1, cax=pst1)
    cb1.ax.tick_params(colors='w')
    pst2 = fig.add_axes([0.92, 0.425, 0.012, 0.33])
    pst2.set_facecolor('k')
    cb2 = fig.colorbar(re2, cax=pst2)
    cb2.ax.tick_params(colors='w')
    #### Save FIG File ####
    pdf.savefig(facecolor='k')
    plt.close()

def plotraw(fin1, fin2, smaples, filname, avg, freqavg, totalch, header, totalsm, choff_low, choff_high, pdf):
    #### Resize data ####    
    if smaples > 1000:
        plotavg = smaples//1000
        smaples = 1000
        fn  = fin1[: plotavg*smaples, :].reshape(smaples, plotavg, -1).mean(axis=1)
        fn2 = fin2[: plotavg*smaples, :].reshape(smaples, plotavg, -1).mean(axis=1)
    else:
        plotavg = 1
        fn = fin1
        fn2 = fin2
    #### Read Para ####
    if header['foff'] < 0:
        ymax = header['fch1']
        ymin = header['fch1'] + header['foff']*header['nchans']
        ymax2 = header['fch1'] + header['foff']*choff_high*freqavg   
        ymin2 = header['fch1'] + header['foff']*header['nchans'] - header['foff']*choff_low*freqavg
    else:
        ymax = header['fch1'] + header['foff']*header['nchans']
        ymin = header['fch1']     
        ymax2 = header['fch1'] + header['foff']*header['nchans'] - header['foff']*choff_high*freqavg
        ymin2 = header['fch1'] + header['foff']*choff_low*freqavg   
    smpt = header['tsamp']*1e6
    freqctr = (ymax+ymin)/2
    if header['telescope_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 64, 65]:
        telescope_id = readfil.ids_to_telescope[header['telescope_id']]
    else:
        telescope_id = 'Unknow'
    if header['machine_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 20, 64, 65]:
        machine_id = readfil.ids_to_machine[header['machine_id']]
    else:
        machine_id = 'Unknow'
    #### Plot ####
    fig, axes = plt.subplots(3, 2, gridspec_kw={'height_ratios':[1,3,3], 'width_ratios':[10, 1]})
    fig.set_facecolor('k')
    fig.set_size_inches(15, 10)
    plt.subplots_adjust(wspace= 0.01, hspace= 0.02, left=0.06, bottom=0.07, right=0.95, top=0.88)
    #### Plot TEXT ####
    fig.suptitle("SHAO_FRB results for:  %s\
    Average:%3d %sFreqAvg:%3d %sBeam: %3d/%3d\n\n\n\n\n"%(filname.ljust(25,' '),
        avg*plotavg, ' '.ljust(3,' '), freqavg, 
        ' '.ljust(3,' '), header['ibeam'], header['nbeams']), color='1') 
    fig.text(0.2, 0.94, "Source: %s"%header['source_name'], color='1', size=12)
    fig.text(0.2, 0.92, "Telscope: %s"%telescope_id, color='1', size=12)
    fig.text(0.2, 0.90, "Instrument: %s"%machine_id, color='1', size=12)
    fig.text(0.4, 0.94, "RA (J2000): %s"%str(header['src_raj']), color='1', size=12)
    fig.text(0.4, 0.92, "DEC (J2000): %s"%str(header['src_dej']), color='1', size=12)
    fig.text(0.4, 0.90, "MJD(bary): %s"%str(header['tstart']), color='1', size=12)    
    fig.text(0.65, 0.94, "N samples: %s"%str(totalsm), color='1', size=12)
    fig.text(0.65, 0.92, "Sampling time: %s us"%str(smpt), color='1', size=12)
    fig.text(0.65, 0.90, "Freq(ctr): %s MHz"%str(freqctr), color='1', size=12)
    ### SET STYLE ####
    for yl in range(3):
        axes[yl, 1].set_yticks([]) 
        for xl in range(2):
            if yl == 2 or xl != 1 :
                axes[yl, xl].set_facecolor('k')
                axes[yl, xl].tick_params(colors='w')
                axes[yl, xl].spines['right'].set_color('w')
                axes[yl, xl].spines['left'].set_color('w')    
                axes[yl, xl].spines['top'].set_color('w')
                axes[yl, xl].spines['bottom'].set_color('w')
    axes[0,1].set_facecolor('k')
    axes[1,1].set_facecolor('k')
    axes[0,0].set_xticks([])    
    axes[1,0].set_xticks([])     
    axes[2,0].set_xticks(np.arange(0, 1001, 100)*plotavg*avg*smpt*1e-6) 
    #### Plot Flux ####
    axes[0,0].plot(np.arange(0, smaples), (fn2.mean(axis=1)), color='w', linewidth=1)
    axes[0,0].set_ylabel("Flux", color='w')    
    axes[0,0].set_xlim(0, smaples)
    fig.text(0.875, 0.86, 'MAX %4.3f'%(fn2.mean(axis=1).max()), color='1')
    #### Plot Result ####
    re1 = axes[2,0].imshow(np.transpose(fn), aspect = 'auto', origin = 'lower',
            extent = [0, smaples*plotavg*avg*smpt*1e-6, ymin, ymax],
            cmap = 'magma')  # viridis, magma, Blues
    axes[2,0].set_ylabel("Frequency (MHz)", color='w')
    axes[2,0].set_xlabel("Time (s)", color='w')
    axes[2,0].tick_params(colors='w')
    #### Plot Raw data ####
    re2 = axes[1,0].imshow(np.transpose(fn2), aspect = 'auto', origin = 'lower',
            extent = [0, smaples, ymin2, ymax2],
            cmap = 'magma') # viridis, magma, Blues
    axes[1,0].set_ylabel("Frequency (MHz)", color='w')
    axes[1,0].tick_params(colors='w')
    #### Plot Raw Flux ####
    plotwinx(axes[2,1], totalch, fn)
    axes[2,1].set_xlabel("Flux", color='w')
    plotwinx(axes[1,1], totalch-choff_low-choff_high, fn2)
    axes[1,1].set_xlabel("Flux", color='w')
    ### Save FIG File ####
    pdf.savefig(facecolor='k')
    plt.close()