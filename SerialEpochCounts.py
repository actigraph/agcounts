import numpy as np


#dataframe - 1D matrix containing raw data
#frequency - sample frequency of raw data (Hz)
#lfeSelect - 0 for regular trimming (LFE disabled), 1 for allow more noise (LFE enabled)
#epochSeconds - used to compute how many raw samples are used for computing an epoch
def epochGeneration(raw, frequency, lfeSelect, epochSeconds):
    if frequency == 30:
        L, M = 1, 1
    elif frequency == 40:
        L, M = 3, 4
    elif frequency == 50:
         L, M = 3, 5
    elif frequency == 60:
         L, M = 1, 2
    elif frequency == 70:
         L, M = 3, 7
    elif frequency == 80:
         L, M = 3, 8
    elif frequency == 90:
         L, M = 1, 3
    elif frequency == 100:
         L, M = 3, 10
    else:
        L, M = 1, 1

    raw = np.transpose(raw)
    #Allocate memory and upsample by factor L.
    upSampleData = np.zeros((1, int(len(raw[0])*L)))

    for i in range(len(raw[0])):
        upSampleData[0, i*L] = raw[0, i]

    #Allocate memory and then LPF.  LPF is only done at non integer multiples of 30 Hz.
    #This LPF is garbage and does a poor job of atteunuating higher frequencies that need to be
    #rejected.  This is the reason why there is aliasing which causes the "tail" on the epochs.
    
    lpfUpSampleData = np.zeros((1, len(upSampleData[0])))
    if not frequency in [30, 60, 90]:
        lpfUpSampleData = np.zeros((1, int(len(raw[0])*L + 1)))
    
    pi=np.pi #3.1415926535897932385
    pi_FP = pi
    a_FP = pi_FP / ( pi + 2*L )    
    b_FP = ( pi-2*L ) / ( pi + 2*L )
    L_FP = L
    
    if frequency==30 or frequency==60 or frequency==90:
        lpfUpSampleData = upSampleData
    else:
        for i in range(1, len(lpfUpSampleData[0])):
            lpfUpSampleData[0, i] = (a_FP*L_FP)*upSampleData[0, i-1] + (a_FP*L_FP)*upSampleData[0, i-2] - b_FP*lpfUpSampleData[0, i-1]
    
    if not frequency in [30, 60, 90]:
        lpfUpSampleData = lpfUpSampleData[:,1:]
    
    #Then allocate memory and downsample by factor M.  Downsampled data is rounded to 3 decimal places before input
    #into BPF.

    downSampleData = np.zeros((1, int(np.floor(len(raw[0])*L/M))))

    if frequency==30:
        downSampleData = raw
    else:
        for i in range(len(downSampleData[0])):
            downSampleData[0, i] = lpfUpSampleData[0, i*M] 

    downSampleData = np.round(downSampleData*1000) / 1000
    
    #BPF.  There are extraneous coefficients as to match constants in ActiLife.
    #Input data coefficients.
    Icf = np.array([[-0.009341062898525, -0.025470289659360, -0.004235264826105,
                     0.044152415456420, 0.036493718347760, -0.011893961934740,
                     -0.022917390623150, -0.006788163862310, 0.000000000000000]])
            
    #Output data coefficients.
    Ocf = np.array([[1.00000000000000000000 ,-3.63367395910957000000 ,5.03689812757486000000 ,
                     -3.09612247819666000000, 0.50620507633883000000, 0.32421701566682000000, 
                     -0.15685485875559000000, 0.01949130205890000000, 0.00000000000000000000]])
    
   
    bpfData = np.zeros((1, len(downSampleData[0])))
    
    shiftRegIn  = np.zeros((1, 9))
    shiftRegOut = np.zeros((1, 9))

    for _ in range(180*6):   #charge filter up to steady state
        shiftRegIn[[0], 1:9] = shiftRegIn[[0], 0:(9-1)]
        shiftRegIn[0, 0]   = downSampleData[0, 0]
        zerosComp = np.sum(Icf[[0], 0:8]*shiftRegIn[[0], 0:8])
        polesComp = np.sum(Ocf[[0], 1:8]*shiftRegOut[[0], 0:7])
        bpfData[0,0] = zerosComp - polesComp
        shiftRegOut[[0], 1:9] = shiftRegOut[[0], 0:(9-1)]
        shiftRegOut[0,0] = zerosComp - polesComp

    for j in range(len(bpfData[0])):
        shiftRegIn[[0], 1:9] = shiftRegIn[[0], 0:8]
        shiftRegIn[0, 0]   = downSampleData[0, j]
        zerosComp = np.sum(Icf[[0], 0:8]*shiftRegIn[[0], 0:8])
        polesComp = np.sum(Ocf[[0], 1:8]*shiftRegOut[[0], 0:7])
        bpfData[0, j] = zerosComp - polesComp
        shiftRegOut[[0], 1:9] = shiftRegOut[[0], 0:(9-1)]
        shiftRegOut[0,0] = zerosComp - polesComp

    bpfData = ((3.0 / 4096.0) / (2.6 / 256.0) * 237.5) * bpfData  #17.127404 is used in ActiLife and 17.128125 is used in firmware.
    
    #then threshold/trim
    trimData = np.zeros( (1, len(bpfData[0])) )
    
    if abs(lfeSelect) > 0:
        MIN_COUNT =   1
        MAX_COUNT = 128*1
        
        for i in range(len(bpfData[0])):
            if abs(bpfData[0, i]) > MAX_COUNT:
                trimData[0,i] = MAX_COUNT
            elif abs(bpfData[0, i]) < MIN_COUNT:
                trimData[0, i] = 0
            elif abs(bpfData[0, i]) < 4:
                trimData[0, i] = np.floor(abs(bpfData[0, i])) - 1
            else:
                trimData[0, i] = np.floor(abs(bpfData[0, i]))  #floor
    else:
        MIN_COUNT =   4
        MAX_COUNT = 128

        for i in range(len(bpfData[0])):
            if abs(bpfData[0, i]) > MAX_COUNT:
                trimData[0, i] = MAX_COUNT
            elif abs(bpfData[0, i])<MIN_COUNT:
                trimData[0, i] = 0
            else:
                trimData[0, i] = np.floor( abs(bpfData[0, i]))  #floor
    
    #hackish downsample to 10 Hz
    downSample10Hz = np.zeros( (1, int(len(trimData[0])/3)) )
  
    for y in range(1, len(downSample10Hz[0]) + 1):
        downSample10Hz[0, y - 1] =  np.floor(np.nanmean(trimData[0, ((y-1)*3):((y-1)*3+3)]))  #floor
    
    #Accumulator for epoch
    blockSize = epochSeconds * 10
    epochCounts = np.zeros( (1, int((len(downSample10Hz[0])/blockSize))))
  
    for i in range(len(epochCounts[0])):
        epochCounts[0, i] = np.floor(sum(downSample10Hz[0, i*blockSize:i*blockSize+blockSize]))



    return epochCounts


def get_counts(rawMatrix, freq, epoch):
    """
    Function for generating counts from raw data.

    Parameters
    ----------
    rawMatrix : ndarray, shape (n_samples, 3)
        Raw data matrix, in x, y, z directions for 1st, 2nd, 3rd columns.
    freq : int
        Sampling frequency, has to be 30, 40, 50, 60, 70, 80, 90 or 100 Hz.
    epoch : bool
        Epoch length (seconds).

    Returns
    -------
    counts : ndarray, shape (n_epochs, 3)
        The counts, n_epochs = ceil(n_samples/freq).    
    """
    
    assert freq in range(30,101,10), "freq must be in [30 : 10 : 100]"
    
    #generating counts for x, y, z 
    xRaw = rawMatrix[0:len(rawMatrix), [0]]
    yRaw = rawMatrix[0:len(rawMatrix), [1]]
    zRaw = rawMatrix[0:len(rawMatrix), [2]]
    epochCountsX = epochGeneration(xRaw, freq, 0, epoch)
    epochCountsY = epochGeneration(yRaw, freq, 0, epoch)
    epochCountsZ = epochGeneration(zRaw, freq, 0, epoch)

    #formatting matrix for output
    xCountsTransposed = np.transpose(epochCountsX)
    yCountsTransposed = np.transpose(epochCountsY)
    zCountsTransposed = np.transpose(epochCountsZ)
    intermediaryMatrix = np.append((xCountsTransposed), yCountsTransposed, 1)
    counts = np.append((intermediaryMatrix), zCountsTransposed, 1)

    return counts.astype(int)

