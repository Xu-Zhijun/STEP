# STEP (SHAO TransiEnts Pipeline)
V1.0 plot pulses in .fil file for giving offset times and DM.

# Require
python3.6+

numpy

matplotlib

# Setting
Modify the frbcfg.ini to set the parameter:

SearchPath  # directory for .fil files

PlotReady   # 1 for Plot, 0 (no function yet)

PLOTFILE    # only has one .fil file to plot, or comment or delete this line.  

PlotBoxcar  # data smooth

PlotTime    # offset sec for each pulses

Plotrange   # (no function yet)

PlotDM      # set DM for pulses

AVERAGE     # Downsample

FREQAVG     # sub-band

CHOFF_LOW   # remove channels from low frequency edge

CHOFF_HIGH  # remove channels from high frequency edge

WINDOWSIZE  # window size for searching, will change the SNR

# Running
python step_plotraw.py

# HAVE FUN!

# Acknowledgment
Thanks scott ransom for presto sigproc.py, github: https://github.com/scottransom/presto
