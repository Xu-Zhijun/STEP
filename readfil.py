#!/usr/bin/env python
"""
Modified from Presto sigproc.py, thanks presto author scott ransom!
Presto github: https://github.com/scottransom/presto
Main modification in line:
    strval = str(filfile.read(strlen), encoding = "utf-8")
"""
import os
import struct
import sys
import warnings

telescope_ids = {"Fake": 0, "Arecibo": 1, "ARECIBO 305m": 1, 
                 "Ooty": 2, "Nancay": 3, "Parkes": 4, "Jodrell": 5,
                 "GBT": 6, "GMRT": 7, "Effelsberg": 8, "ATA": 9,
                 "SRT": 10, "LOFAR": 11, "VLA": 12, "CHIME": 20,
                 "FAST": 21, "MeerKAT": 64, "KAT-7": 65}
ids_to_telescope = dict(zip(telescope_ids.values(), telescope_ids.keys()))

machine_ids = {"FAKE": 0, "PSPM": 1, "Wapp": 2, "WAPP": 2, "AOFTM": 3,
               "BCPM1": 4, "BPP": 4, "OOTY": 5, "SCAMP": 6,
               "GBT Pulsar Spigot": 7, "SPIGOT": 7, "BG/P": 11,
               "PDEV": 12, "CHIME+PSR": 20, "KAT": 64, "KAT-DC2": 65}
ids_to_machine = dict(zip(machine_ids.values(), machine_ids.keys()))

header_params = {
    "HEADER_START": 'flag',
    "telescope_id": 'i',
    "machine_id": 'i',
    "data_type": 'i', 
    "rawdatafile": 'str',
    "source_name": 'str', 
    "barycentric": 'i', 
    "pulsarcentric": 'i', 
    "az_start": 'd',  
    "za_start": 'd',  
    "src_raj": 'd',  
    "src_dej": 'd',  
    "tstart": 'd',  
    "tsamp": 'd',  
    "nbits": 'i', 
    "signed": 'b', 
    "nsamples": 'i', 
    "nbeams": "i",
    "ibeam": "i",
    "fch1": 'd',  
    "foff": 'd',
    "FREQUENCY_START": 'flag',
    "fchannel": 'd',  
    "FREQUENCY_END": 'flag',
    "nchans": 'i', 
    "nifs": 'i', 
    "refdm": 'd',  
    "period": 'd',  
    "npuls": 'q',
    "nbins": 'i', 
    "HEADER_END": 'flag'}

def read_header(infile):
    """
    read_header(infile):
       Read a SIGPROC-style header and return the keys/values in a dictionary,
          as well as the length of the header: (hdrdict, hdrlen)
    """
    hdrdict = {"ibeam": 1, "nbeams":1,}
    hdrlen = 0
    if type(infile) == type("abc"):
        # infile = open(infile, 'rb')
        with open(infile, 'rb') as infile:
            param = ""
            while (param != "HEADER_END"):
                param, val = read_hdr_val(infile, stdout=False)
                hdrdict[param] = val
            hdrlen = infile.tell()
    # infile.close()
    return hdrdict, hdrlen

def read_hdr_val(filfile, stdout=False):
    paramname = read_paramname(filfile, stdout)
    try:
        if header_params[paramname] == 'd':
            return paramname, read_doubleval(filfile, stdout)
        elif header_params[paramname] == 'i':
            return paramname, read_intval(filfile, stdout)
        elif header_params[paramname] == 'q':
            return paramname, read_longintval(filfile, stdout)
        elif header_params[paramname] == 'b':
            return paramname, read_charval(filfile, stdout)
        elif header_params[paramname] == 'str':
            return paramname, read_string(filfile, stdout)
        elif header_params[paramname] == 'flag':
            return paramname, None
    except KeyError:
        warnings.warn("key '%s' is unknown!" % paramname)
        return None, None

def read_paramname(filfile, stdout=False):
    paramname = read_string(filfile, stdout=False)
    if stdout:
        print("Read '%s'"%paramname)
    return paramname

def read_doubleval(filfile, stdout=False):
    dblval = struct.unpack('d', filfile.read(8))[0]
    if stdout:
        print("  double value = '%20.15f'"%dblval)
    return dblval

def read_intval(filfile, stdout=False):
    intval = struct.unpack('i', filfile.read(4))[0]
    if stdout:
        print("  int value = '%d'"%intval)
    return intval

def read_charval(filfile, stdout=False):
    charval = struct.unpack('b', filfile.read(1))[0]
    if stdout:
        print(" char value = '%d'"%charval)
    return charval

def read_longintval(filfile, stdout=False):
    longintval = struct.unpack('q', filfile.read(8))[0]
    if stdout:
        print("  long int value = '%d'"%longintval)
    return longintval

def read_string(filfile, stdout=False):
    strlen = struct.unpack('i', filfile.read(4))[0]
    strval = str(filfile.read(strlen), encoding = "utf-8") # Main modification
    if stdout:
        print("  string = '%s'"%strval)
    return strval

def samples_per_file(infile, hdrdict, hdrlen):
    """
    samples_per_file(infile, hdrdict, hdrlen):
       Given an input SIGPROC-style filterbank file and a header
           dictionary and length (as returned by read_header()),
           return the number of (time-domain) samples in the file.
    """
    numbytes = os.stat(infile)[6] - hdrlen
    bytes_per_sample = hdrdict['nchans'] * (hdrdict['nbits']/8)
    if numbytes % bytes_per_sample:
        print("Warning!:  File does not appear to be of the correct length!")
    numsamples = numbytes / bytes_per_sample
    return int(numsamples)