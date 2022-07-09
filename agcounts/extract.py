"""Function for extracting counts."""
import gc
from typing import Any

import numpy as np
from numpy import typing as npt
from scipy import signal

from agcounts.pow2 import resample_to_30hz

# BPF. There are extraneous coefficients as to match constants
# in ActiLife.
# Input data coefficients.
INPUT_COEFFICIENTS: npt.NDArray[np.float_] = np.array(
    [
        [
            -0.009341062898525,
            -0.025470289659360,
            -0.004235264826105,
            0.044152415456420,
            0.036493718347760,
            -0.011893961934740,
            -0.022917390623150,
            -0.006788163862310,
            0.000000000000000,
        ]
    ]
)

# Output data coefficients.
OUTPUT_COEFFICIENTS: npt.NDArray[np.float_] = np.array(
    [
        [
            1.00000000000000000000,
            -3.63367395910957000000,
            5.03689812757486000000,
            -3.09612247819666000000,
            0.50620507633883000000,
            0.32421701566682000000,
            -0.15685485875559000000,
            0.01949130205890000000,
            0.00000000000000000000,
        ]
    ]
)


def _factors(frequency: int):
    if frequency == 30:
        upsample_factor, downsample_factor = 1, 1
    elif frequency == 40:
        upsample_factor, downsample_factor = 3, 4
    elif frequency == 50:
        upsample_factor, downsample_factor = 3, 5
    elif frequency == 60:
        upsample_factor, downsample_factor = 1, 2
    elif frequency == 70:
        upsample_factor, downsample_factor = 3, 7
    elif frequency == 80:
        upsample_factor, downsample_factor = 3, 8
    elif frequency == 90:
        upsample_factor, downsample_factor = 1, 3
    elif frequency == 100:
        upsample_factor, downsample_factor = 3, 10
    else:
        upsample_factor, downsample_factor = 1, 1
    return upsample_factor, downsample_factor


def _extract_slow(
    raw: npt.NDArray[np.float_],
    frequency: int,
    lfe_select: bool,
    epoch: int,
    verbose: bool,
):
    """Extract actigraphy counts from one-dimensional accelerometer data.

    This implementation is purposefully only using basic python function to allow
    easier understanding of the algorithm.

    Parameters
    ----------
    raw
        1D matrix containing raw data
    frequency
        sample frequency of raw data (Hz)
    lfe_select
        False for regular trimming (LFE disabled), True to allow more noise
        (LFE enabled)
    epoch
        Epoch duration in seconds

    Returns
    -------
    ndarray containing epoch counts
    """
    upsample_factor, downsample_factor = _factors(frequency)
    raw = np.transpose(raw)
    # Allocate memory and upsample by factor L.
    upsample_data = np.zeros((1, int(len(raw[0]) * upsample_factor)))

    for i in range(len(raw[0])):
        upsample_data[0, i * upsample_factor] = raw[0, i]

    # Allocate memory and then LPF.  LPF is only done at non integer multiples of 30 Hz.
    # This LPF is garbage and does a poor job of attenuating higher frequencies that
    # need to be rejected.  This is the reason why there is aliasing which causes the
    # "tail" on the epochs.

    if frequency not in [30, 60, 90]:
        lpf_upsample_data = np.zeros((1, int(len(raw[0]) * upsample_factor + 1)))

    pi = np.pi  # 3.1415926535897932385
    pi_fp = pi
    a_fp = pi_fp / (pi + 2 * upsample_factor)
    b_fp = (pi - 2 * upsample_factor) / (pi + 2 * upsample_factor)
    l_fp = upsample_factor

    if verbose:
        print("Upsampling data")
    if frequency == 30 or frequency == 60 or frequency == 90:
        lpf_upsample_data = upsample_data
    else:
        for i in range(1, len(lpf_upsample_data[0])):
            lpf_upsample_data[0, i] = (
                (a_fp * l_fp) * upsample_data[0, i - 1]
                + (a_fp * l_fp) * upsample_data[0, i - 2]
                - b_fp * lpf_upsample_data[0, i - 1]
            )

    if frequency not in [30, 60, 90]:
        lpf_upsample_data = lpf_upsample_data[:, 1:]

    # Then downsample by factor M.  Downsampled data is rounded to 3 decimal places
    # before input into BPF.
    if verbose:
        print("Downsampling data")
    if frequency == 30:
        down_sample_data = raw
    else:
        down_sample_data = np.zeros(
            (1, int(np.floor(len(raw[0]) * upsample_factor / downsample_factor)))
        )
        for i in range(len(down_sample_data[0])):
            down_sample_data[0, i] = lpf_upsample_data[0, i * downsample_factor]
    del raw
    del lpf_upsample_data
    gc.collect(1)

    down_sample_data = np.round(down_sample_data * 1000) / 1000

    bpf_data: npt.NDArray[np.float_] = np.zeros((1, len(down_sample_data[0])))

    shift_reg_in = np.zeros((1, 9))
    shift_reg_out = np.zeros((1, 9))

    if verbose:
        print("Filtering data")
    for _ in range(180 * 6):  # charge filter up to steady state
        shift_reg_in[[0], 1:9] = shift_reg_in[[0], 0 : (9 - 1)]
        shift_reg_in[0, 0] = down_sample_data[0, 0]
        zeros_comp = np.sum(INPUT_COEFFICIENTS[[0], 0:8] * shift_reg_in[[0], 0:8])
        poles_comp = np.sum(OUTPUT_COEFFICIENTS[[0], 1:8] * shift_reg_out[[0], 0:7])
        bpf_data[0, 0] = zeros_comp - poles_comp
        shift_reg_out[[0], 1:9] = shift_reg_out[[0], 0 : (9 - 1)]
        shift_reg_out[0, 0] = zeros_comp - poles_comp

    for j in range(len(bpf_data[0])):
        shift_reg_in[[0], 1:9] = shift_reg_in[[0], 0:8]
        shift_reg_in[0, 0] = down_sample_data[0, j]
        zeros_comp = np.sum(INPUT_COEFFICIENTS[[0], 0:8] * shift_reg_in[[0], 0:8])
        poles_comp = np.sum(OUTPUT_COEFFICIENTS[[0], 1:8] * shift_reg_out[[0], 0:7])
        bpf_data[0, j] = zeros_comp - poles_comp
        shift_reg_out[[0], 1:9] = shift_reg_out[[0], 0 : (9 - 1)]
        shift_reg_out[0, 0] = zeros_comp - poles_comp
    del down_sample_data
    gc.collect(1)

    bpf_data = (
        (3.0 / 4096.0) / (2.6 / 256.0) * 237.5
    ) * bpf_data  # 17.127404 is used in ActiLife and 17.128125 is used in firmware.

    if verbose:
        print("Threshold/trimming data")

    # then threshold/trim
    trim_data = np.zeros((1, len(bpf_data[0])))  # type: ignore

    if lfe_select:
        min_count = 1
        max_count = 128 * 1

        for i in range(len(bpf_data[0])):  # type: ignore
            if abs(bpf_data[0, i]) > max_count:  # type: ignore
                trim_data[0, i] = max_count
            elif abs(bpf_data[0, i]) < min_count:  # type: ignore
                trim_data[0, i] = 0
            elif abs(bpf_data[0, i]) < 4:  # type: ignore
                trim_data[0, i] = np.floor(abs(bpf_data[0, i])) - 1  # type: ignore
            else:
                trim_data[0, i] = np.floor(abs(bpf_data[0, i]))  # type: ignore
    else:
        min_count = 4
        max_count = 128

        for i in range(len(bpf_data[0])):  # type: ignore
            if abs(bpf_data[0, i]) > max_count:  # type: ignore
                trim_data[0, i] = max_count
            elif abs(bpf_data[0, i]) < min_count:  # type: ignore
                trim_data[0, i] = 0
            else:
                trim_data[0, i] = np.floor(abs(bpf_data[0, i]))  # type: ignore

    if verbose:
        print("Getting data back to 10Hz for accumulation")
    del bpf_data
    # hackish downsample to 10 Hz
    down_sample10_hz = np.zeros((1, int(len(trim_data[0]) / 3)))

    for y in range(1, len(down_sample10_hz[0]) + 1):
        down_sample10_hz[0, y - 1] = np.floor(
            np.nanmean(trim_data[0, ((y - 1) * 3) : ((y - 1) * 3 + 3)])
        )  # floor
    del trim_data
    gc.collect(1)

    # Accumulator for epoch
    block_size = epoch * 10
    epoch_counts = np.zeros((1, int((len(down_sample10_hz[0]) / block_size))))

    if verbose:
        print("Summing epochs")
    for i in range(len(epoch_counts[0])):
        epoch_counts[0, i] = np.floor(
            sum(down_sample10_hz[0, i * block_size : i * block_size + block_size])
        )
    del down_sample10_hz
    gc.collect(1)
    return epoch_counts


def _resample(
    raw: npt.NDArray[np.float_], frequency: int, verbose: bool
) -> npt.NDArray[np.float64]:
    """Resample the data.

    Parameters
    ----------
    raw:
        Matrix containing raw data
    frequency:
        sample frequency of raw data (Hz)
    verbose:
        Print diagnostic messages

    Returns
    -------
    resampled_data :
        The resampled_data
    """
    upsample_factor, downsample_factor = _factors(frequency)
    raw = np.transpose(raw)

    # Upsample by factor L.
    m, n = raw.shape
    upsample_data = np.zeros((m, upsample_factor * n), dtype=raw.dtype)
    upsample_data[:, ::upsample_factor] = raw

    pi = np.pi  # 3.1415926535897932385
    a_fp = pi / (pi + 2 * upsample_factor)
    b_fp = (pi - 2 * upsample_factor) / (pi + 2 * upsample_factor)
    up_factor_fp = upsample_factor

    # raw doesn't need to be used after this
    if frequency != 30:
        del raw
    gc.collect(1)

    # Allocate memory and then LPF.  LPF is only done at non
    # integer multiples of 30 Hz. This LPF is garbage and does a
    # poor job of attenuating higher frequencies that need to be
    # rejected. This is the reason why there is aliasing which
    # causes the "tail" on the epochs.
    if frequency not in [30, 60, 90]:
        upsample_data = (a_fp * up_factor_fp) * (
            upsample_data + np.roll(upsample_data, 1)
        )
        z = np.zeros(upsample_data.shape[0])
        upsample_data = np.column_stack((z, upsample_data))
        del z

        for i in range(1, len(upsample_data[0])):
            upsample_data[:, i] += -b_fp * upsample_data[:, i - 1]
        # remove the zeros
        upsample_data = upsample_data[:, 1:]

    # Then allocate memory and downsample by factor M. Downsampled
    # data is rounded to 3 decimal places before input into BPF.
    if frequency == 30:
        downsample_data = raw
        del raw
        gc.collect(1)
    else:
        downsample_data = upsample_data[:, ::downsample_factor]

    del upsample_data
    gc.collect(1)
    if verbose:
        print("Created downsample_data")
    downsample_data = np.round(downsample_data * 1000) / 1000
    return downsample_data


def _bpf_filter(downsample_data: npt.NDArray[np.float_], verbose: bool) -> Any:
    """Run BPF Filter.

    Parameters
    ----------
    downsample_data:
        Matrix containing downsampled data
    verbose:
        Print diagnostic messages

    Returns
    -------
    bpf_data :
        The filtered data
    """
    zi = signal.lfilter_zi(INPUT_COEFFICIENTS[0, :], OUTPUT_COEFFICIENTS[0, :]).reshape(
        (1, -1)
    )
    if verbose:
        print("Filtering Data")
    bpf_data, _ = signal.lfilter(
        INPUT_COEFFICIENTS[0, :],
        OUTPUT_COEFFICIENTS[0, :],
        downsample_data,
        zi=zi.repeat(downsample_data.shape[0], axis=0)
        * downsample_data[:, 0].reshape((-1, 1)),
    )

    del downsample_data
    gc.collect(1)

    bpf_data = ((3.0 / 4096.0) / (2.6 / 256.0) * 237.5) * bpf_data
    # 17.127404 is used in ActiLife and 17.128125 is used in
    # firmware.
    return bpf_data


def _trim_data(
    bpf_data: npt.NDArray[np.float_], lfe_select: bool, verbose: bool
) -> Any:
    """Trim/Threshold data.

    Parameters
    ----------
    bpf_data:
        Matrix containing filtered data
    lfe_select:
        False for regular trimming, True for allow more noise
    verbose:
        Print diagnostic messages

    Returns
    -------
    trim_data :
        The trimmed/thresholded data
    """
    # then threshold/trim
    if verbose:
        print("Trimming Data")
    if lfe_select:
        min_count = 1
        max_count = 128 * 1

        trim_data = np.abs(bpf_data)
        trim_data[(trim_data < min_count) & (trim_data >= 4)] = 0
        trim_data[trim_data > max_count] = max_count
        mask = (trim_data < 4) & (trim_data >= min_count)
        trim_data[mask] = np.abs(trim_data[mask]) - 1
        trim_data = np.floor(trim_data)
        del mask

    else:
        min_count = 4
        max_count = 128

        trim_data = np.abs(bpf_data)
        trim_data[trim_data < min_count] = 0
        trim_data[trim_data > max_count] = max_count
        trim_data = np.floor(trim_data)
    return trim_data


def _resample_10hz(
    trim_data: npt.NDArray[np.float_], verbose: bool
) -> npt.NDArray[np.float64]:
    """Resample the data.

    Parameters
    ----------
    trim_data:
        Matrix containing trimmed/thresholded data


    Returns
    -------
    resampled_data :
        The resampled_data
    """
    if verbose:
        print("Getting data back to 10Hz for accumulation")
    # hackish downsample to 10 Hz
    downsample_10hz = np.cumsum(trim_data, axis=-1, dtype=float)
    downsample_10hz[:, 3:] = downsample_10hz[:, 3:] - downsample_10hz[:, :-3]
    downsample_10hz = np.floor(downsample_10hz[:, 3 - 1 :: 3] / 3)
    return downsample_10hz


def _sum_counts(
    downsample_10hz: npt.NDArray[np.float_], epoch_seconds: int, verbose: bool
) -> npt.NDArray[np.float64]:
    """Generate counts.

    Parameters
    ----------
    downsample_10hz:
        Matrix containing downsampled to 10hz data
    epoch_seconds:
        Used to compute how many raw samples are used for
        computing an epoch
    verbose:
        Print diagnostic messages

    Returns
    -------
    epochs :
        The epochs
    """
    if verbose:
        print("Summing epochs")
    # Accumulator for epoch
    block_size = epoch_seconds * 10
    epoch_counts = np.cumsum(downsample_10hz, axis=-1, dtype=float)

    epoch_counts[:, block_size:] = (
        epoch_counts[:, block_size:] - epoch_counts[:, :-block_size]
    )
    epoch_counts = np.floor(epoch_counts[:, block_size - 1 :: block_size])
    return epoch_counts


def _extract(
    raw: npt.NDArray[np.float_],
    frequency: int,
    lfe_select: bool,
    epoch_seconds: int,
    verbose: bool,
) -> npt.NDArray[np.float64]:
    """Generate counts.

    Parameters
    ----------
    raw:
        Matrix containing raw data
    frequency:
        sample frequency of raw data (Hz)
    lfe_select:
        False for regular trimming, True for allow more noise
    epoch_seconds:
        Used to compute how many raw samples are used for
        computing an epoch
    verbose:
        Print diagnostic messages

    Returns
    -------
    epochs :
        The epochs
    """
    downsample_data = _resample(raw=raw, frequency=frequency, verbose=verbose)
    del raw
    bpf_data = _bpf_filter(downsample_data=downsample_data, verbose=verbose)
    del downsample_data
    trim_data = _trim_data(bpf_data=bpf_data, lfe_select=lfe_select, verbose=verbose)
    del bpf_data
    downsample_10hz = _resample_10hz(trim_data=trim_data, verbose=verbose)
    del trim_data
    gc.collect(1)
    epoch_counts = _sum_counts(
        downsample_10hz=downsample_10hz, epoch_seconds=epoch_seconds, verbose=verbose
    )
    return epoch_counts


def get_counts(raw, freq: int, epoch: int, fast: bool = True, verbose: bool = False):
    """
    Generate counts from raw data.

    Parameters
    ----------
    raw : ndarray, shape (n_samples, ANY)
        Raw data matrix
    freq : int
        Sampling frequency, has to be 30, 40, 50, 60, 70, 80, 90 or 100 Hz.
    epoch : int
        Epoch length (seconds).
    fast:
        Use fast implementation
    verbose:
        Print diagnostic messages

    Returns
    -------
    counts : ndarray, shape (n_epochs, ANY)
        The counts, n_epochs = ceil(n_samples/freq).
    """
    if freq in [32, 64, 128, 256]:
        raw = resample_to_30hz(raw, freq)
        freq = 30
    else:
        assert freq in range(30, 101, 10), (
            "freq must be in [30 : 10 : 100]," "or a power of 2 between 32 and 256."
        )

    if fast:
        counts = _extract(raw, freq, False, epoch, verbose > 1).transpose()
    else:
        axis_counts = []
        for i in range(raw.shape[1]):
            print(f"Running extract for axis {i} of {raw.shape[1]}")
            axis_counts.append(_extract_slow(raw[0:, [i]], freq, False, epoch, verbose))

        counts = np.concatenate(axis_counts, 0).transpose()

    return counts.astype(int)
