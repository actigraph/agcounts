import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SerialEpochCounts import get_counts

data_path = r"your-data-path"

# Check epoch lengths 30 and 10 seconds and all admissible sampling frequencies
# (60 and 90 Hz output from ActiLife do not make sense - they are treated just
# like 30 Hz, contact software engineering) for validation purposes

mean = 0
std = 1 
freqs = [30, 40, 50, 60, 70, 80, 90, 100]
n_epochs = 1000
epochs = [30, 10]
for epoch in epochs:
    for freq in freqs:
        # Create white noise
        num_samples = int(n_epochs*freq*epoch)
        time_length = num_samples/freq
        signal_x = np.random.normal(mean, std, size=num_samples)
        signal_y = np.random.normal(mean, std, size=num_samples)
        signal_z = np.random.normal(mean, std, size=num_samples)
        
        # Dump to CSV which can be opened in ActiLife
        signals = np.array([signal_x, signal_y, signal_z]).T
        np.savetxt(data_path+r"\raw_" + str(epoch) + '_' + str(freq) + ".csv", signals, delimiter=",")

#####
# <Copy over the the title info from an ordinary csv raw file into the 
# saved csv files, adjust the sampling frequency and open with ActiLife and 
# convert to counts using raw->epochs functionality>
#####

for epoch in epochs:
    for freq in freqs:
        counts_acti = pd.read_csv(data_path+r"\raw_" + str(epoch) + '_' + str(freq) + str(epoch) + "sec.csv",  skiprows = 10)
        counts_acti_np = np.array(counts_acti)
        counts_acti_np = np.array([counts_acti_np[:, 1], counts_acti_np[:, 0], counts_acti_np[:, 2]]).T # ActiLife switches out the x and y axes, so switch them back for consistency
        signals = pd.read_csv(data_path+r"\raw_" + str(epoch) + '_' + str(freq) + ".csv", skiprows = 11, header=None)
        counts_py_serial = get_counts(np.array(signals), freq=freq, epoch=epoch)
        print('Done')
        if freq not in [60, 90]: # Do not get hung up on 60 and 90 Hz - need to clarify why these are treated as 30 Hz by software engineering
            if (abs(counts_py_serial - counts_acti_np) > 1).any():
                print(counts_py_serial - counts_acti_np)
                print(freq)
                print(epoch)

# Now do more epochs (1000) for 30 and 40 Hz, 30 and 10 seconds epochs.
# These are the data showed in the manuscript
freqs = [30, 40]
n_epochs = 1000
epochs = [30, 10]
for epoch, freq in zip(epochs, freqs):
    # Create white noise
    num_samples = int(n_epochs*freq*epoch)
    time_length = num_samples/freq
    signal_x = np.random.normal(mean, std, size=num_samples)
    signal_y = np.random.normal(mean, std, size=num_samples)
    signal_z = np.random.normal(mean, std, size=num_samples)
    
    # Dump to CSV which can be opened in ActiLife
    signals = np.array([signal_x, signal_y, signal_z]).T
    np.savetxt(data_path+r"\raw_" + str(epoch) + '_' + str(freq) + ".csv", signals, delimiter=",")

#####
# <Copy over the the title info from an ordinary csv raw file into the 
# saved csv files, adjust the sampling frequency and open with ActiLife and 
# convert to counts using raw->epochs functionality>
#####

all_errors = []
freqs = [30, 40] # The actilife output at 60 and 90 Hz does not make sense
n_epochs = 1000
epochs = [30, 10]
for epoch, freq in zip(epochs, freqs):
    counts_acti = pd.read_csv(data_path+r"\raw_" + str(epoch) + '_' + str(freq) + str(epoch) + "sec.csv",  skiprows = 10)
    counts_acti_np = np.array(counts_acti)
    counts_acti_np = np.array([counts_acti_np[:, 1], counts_acti_np[:, 0], counts_acti_np[:, 2]]).T # ActiLife switches out the x and y axes, so switch them back for consistency
    signals = pd.read_csv(data_path+r"\raw_" + str(epoch) + '_' + str(freq) + ".csv", skiprows = 11, header=None)
    counts_py_serial = get_counts(np.array(signals), freq=freq, epoch=epoch)
    counts_py_serial = counts_py_serial.flatten()
    counts_acti_np = counts_acti_np.flatten()
    
    min_length = np.min([counts_py_serial.shape, counts_acti_np.shape])
    errors = counts_py_serial[:min_length] - counts_acti_np[:min_length]
    all_errors.append(errors)
    mean = (counts_py_serial + counts_acti_np) / 2
    mean_abs_rel_errors = np.abs(errors / counts_acti_np * 100)
    np.mean(mean_abs_rel_errors)
    np.std(mean_abs_rel_errors)

    # Bland-altman plot
    font = {'size'   : 8}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.scatter(mean, errors, s=0.5, color='k')
    plt.xlabel('Counts')
    plt.ylabel('Error (counts)')
    plt.axhline(y=np.mean(errors), color='red', linestyle='--', linewidth=1)
    plt.axhline(y=np.mean(errors) + np.std(errors)*1.96, color='orange', linestyle='--', linewidth=1)
    plt.axhline(y=np.mean(errors) - np.std(errors)*1.96, color='orange', linestyle='--', linewidth=1)
    plt.tight_layout()
    
    # Histogram of errors
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.hist(errors, bins=70, color='gray')
    plt.ylabel('Number of epochs')
    plt.xlabel('Error (counts)')
    plt.tight_layout()
    
    # Histogram of mean absolute percentage error
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.hist(mean_abs_rel_errors, color='gray')
    plt.ylabel('Number of epochs')
    plt.xlabel('Absolute percentage error')
    plt.tight_layout()

