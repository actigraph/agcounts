# agcounts
![Tests](https://github.com/actigraph/agcounts/actions/workflows/tests.yml/badge.svg)

A python package for extracting actigraphy counts from accelerometer data. 

```bash
curl -L https://github.com/actigraph/agcounts/files/8247896/GT3XPLUS-AccelerationCalibrated-1x8x0.NEO1G75911139.2000-01-06-13-00-00-000-P0000.sensor.csv.gz --output data.csv.gz
```
```python
import pandas as pd
import numpy as np
from agcounts.extract import get_counts

def get_counts_csv(file, freq: int, epoch: int, fast: bool = True, verbose: bool = False, time_column: str = None):
  if verbose:
    print("Reading in CSV", flush = True)
  raw = pd.read_csv(file, skiprows=0)
  if time_column is not None:
    ts = raw[time_column]
    ts = pd.to_datetime(ts)
    time_freq = str(epoch) + 'S'
    ts = ts.dt.round(time_freq)
    ts = ts.unique()
    ts = pd.DataFrame(ts, columns=[time_column])
  raw = raw[["X", "Y", "Z"]]
  if verbose:
    print("Converting to array", flush = True)  
  raw = np.array(raw)
  if verbose:
    print("Getting Counts", flush = True)    
  counts = get_counts(raw, freq = freq, epoch = epoch, fast = fast, verbose = verbose)
  del raw
  counts = pd.DataFrame(counts, columns = ['X','Y','Z'])
  ts = ts[0:counts.shape[0]]
  if time_column is not None:
    counts = pd.concat([ts, counts], axis=1)
  return counts

def convert_counts_csv(file, outfile, freq: int, epoch: int, fast: bool = True, verbose: bool = False, time_column: str = None):
  counts = get_counts_csv(file, freq = 80, epoch = 60, verbose = True, time_column = time_column)
  counts.to_csv(outfile)
  return counts

counts = get_counts_csv("data.csv.gz", freq = 80, epoch = 60)
counts = convert_counts_csv("data.csv.gz", outfile="counts.csv.gz", 
freq=80, epoch = 60, verbose = True, time_column = "HEADER_TIMESTAMP")
```
