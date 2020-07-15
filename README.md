# STEP (SHAO TransiEnts Pipeline)
The goal of this **STEP** is to create a TransiEnts Pipeline based on **CSKAP**(**C**hina **SKA** Regional Centre **P**rototype) system.

The pipeline is written in pure python for flexibility.

## Credit

If you used this software in your search, please cite [@Xu-Zhijun](https://www.github.com/Xu-Zhijun) .

## Contributing

Please feel free to contact [@Xu-Zhijun](https://www.github.com/Xu-Zhijun/STEP) for opening a new issue for bugs, feedback or feature requests.

We welcome code contribution. To add a contribution, please submit a PR.

## Installation

Just use the following command to install the STEP,

```bash
$ pip install -r requirments.txt
```

We recommend to use virtualenv to create a virtual python enviroment.


## Requirements

**STEP** need the following software:

- python3.6+
- numpy
- matplotlib



## How to Run

### Run in one command

```bash
$ python step_plotraw.py
```

In this way you shoulb modify the following configuration in `frbcfg.ini` file.

Modify the frbcfg.ini to set the parameter:

```
SearchPath  # directory for .fil files

PlotReady   # 1 for Plot, 0 (no function yet)

PLOTFILE    # only has one .fil file to plot, or comment or delete this line.  

PlotBoxcar  # data smooth

PlotTime    # offset sec for each pulses

Plotrange   # scale the plot size

PlotDM      # set DM for pulses

PlotPersent # set the persent number you want to plot, like 0.05 only plot top 5% of the data.

AVERAGE     # Downsample

FREQAVG     # sub-band

CHOFF_LOW   # remove channels from low frequency edge

CHOFF_HIGH  # remove channels from high frequency edge

WINDOWSIZE  # window size for searching, will change the SNR
```

or by using command paramters

### Run by using command paramters

TBA

# Acknowledgment

Thanks scott ransom for [presto](https://github.com/scottransom/presto) sigproc.py .

And HAVE FUN with **STEP**.

