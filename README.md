# agcounts
![Tests](https://github.com/actigraph/agcounts/actions/workflows/tests.yml/badge.svg)

A python package for extracting actigraphy counts from accelerometer data. 

## Install
```bash
pip install agcounts
```
## Test
Download test data:
```bash
curl -L https://github.com/actigraph/agcounts/files/8247896/GT3XPLUS-AccelerationCalibrated-1x8x0.NEO1G75911139.2000-01-06-13-00-00-000-P0000.sensor.csv.gz --output data.csv.gz
```

Run a simple test
```python
from pandas import read_csv
import numpy as np
from agcounts.extract import get_counts

def get_counts_csv(file, freq: int, epoch: int, fast: bool = True, verbose: bool = False):
  if verbose:
    print("Reading in CSV", flush = True)
  raw = read_csv(file, skiprows=0)
  raw = raw[["X", "Y", "Z"]]
  if verbose:
    print("Converting to array", flush = True)  
  raw = np.array(raw)
  if verbose:
    print("Getting Counts", flush = True)    
  counts = get_counts(raw, freq = freq, epoch = epoch, fast = fast, verbose = verbose)
  del raw
  return counts

counts = get_counts_csv("data.csv.gz", freq = 80, epoch = 60)
```