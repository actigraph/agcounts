"""Function for extracting counts."""

import gc
import logging
from typing import Any

import numpy as np
from numpy import typing as npt
from scipy import signal

from agcounts.legacy import (
    INPUT_COEFFICIENTS,
    OUTPUT_COEFFICIENTS,
    _extract_slow,
    _factors,
)

try:
    from mne import filter as mne_filter
except ImportError:
    mne_filter = None

from agcounts.pow2 import resample_to_30hz

logger = logging.getLogger(__name__)


def _resample(
    raw: npt.NDArray[np.float_],
    frequency: int,
    use_mne_filter: bool = False,
) -> Any:
    """Resample the data.

    Parameters
    ----------
    raw:
        Matrix containing raw data
    frequency:
        sample frequency of raw data (Hz)
    use_mne_filter:
        Resample data to 30Hz using MNE's implementation of low pass filter and resample
        instead of ActiGraph's.

    Returns
    -------
    resampled_data :
        The resampled_data

    """
    raw = np.transpose(raw)
    if frequency == 30:
        return np.round(raw, decimals=3)
    if use_mne_filter:
        if mne_filter is None:
            logger.error("Interpolating needs MNE. Please install MNE.")
            raise ImportError("mne")
        else:
            downsample_data = mne_filter.resample(raw, down=frequency / 30)
    else:
        upsample_factor, downsample_factor = _factors(frequency)

        # Upsample by factor L.
        m, n = raw.shape
        upsample_data = np.zeros((m, upsample_factor * n), dtype=raw.dtype)
        upsample_data[:, ::upsample_factor] = raw

        pi = np.pi  # 3.1415926535897932385
        a_fp = pi / (pi + 2 * upsample_factor)
        b_fp = (pi - 2 * upsample_factor) / (pi + 2 * upsample_factor)
        up_factor_fp = upsample_factor

        # Allocate memory and then LPF.  LPF is only done at non
        # integer multiples of 30 Hz. This LPF does a poor job of
        # attenuating higher frequencies that need to be rejected.
        # This is the reason why there is aliasing which causes the
        # "tail" on the epochs.
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
    logger.info("Created downsample_data")
    downsample_data = np.round(downsample_data, decimals=3)
    return downsample_data


def _bpf_filter(downsample_data: npt.NDArray[np.float_]) -> Any:
    """Run BPF Filter.

    Parameters
    ----------
    downsample_data:
        Matrix containing downsampled data

    Returns
    -------
    bpf_data :
        The filtered data
    """
    zi = signal.lfilter_zi(INPUT_COEFFICIENTS[0, :], OUTPUT_COEFFICIENTS[0, :]).reshape(
        (1, -1)
    )
    logger.info("Filtering Data")
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


def _trim_data(bpf_data: npt.NDArray[np.float_], lfe_select: bool) -> Any:
    """Trim/Threshold data.

    Parameters
    ----------
    bpf_data:
        Matrix containing filtered data
    lfe_select:
        False for regular trimming, True for allow more noise

    Returns
    -------
    trim_data :
        The trimmed/thresholded data
    """
    # then threshold/trim
    logger.info("Trimming Data")
    if lfe_select:
        min_count = 1
        max_count = 128 * 1

        trim_data = np.abs(bpf_data)
        mask_larger_than_max = trim_data > max_count
        mask_smaller_than_min = trim_data < min_count
        mask_smaller_than_4 = (trim_data < 4) & (trim_data >= min_count)
        mask_other = ~(
            mask_smaller_than_4 | mask_smaller_than_min | mask_larger_than_max
        )

        trim_data[mask_larger_than_max] = max_count
        trim_data[mask_smaller_than_min] = 0
        trim_data[mask_smaller_than_4] = np.floor(trim_data) - 1
        trim_data[mask_other] = np.floor(trim_data)
    else:
        min_count = 4
        max_count = 128

        trim_data = np.abs(bpf_data)
        trim_data[trim_data < min_count] = 0
        trim_data[trim_data > max_count] = max_count
        trim_data = np.floor(trim_data)
    return trim_data


def _resample_10hz(
    trim_data: npt.NDArray[np.float_],
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
    logger.info("Getting data back to 10Hz for accumulation")
    # hackish downsample to 10 Hz
    downsample_10hz = np.cumsum(trim_data, axis=-1, dtype=float)
    downsample_10hz[:, 3:] = downsample_10hz[:, 3:] - downsample_10hz[:, :-3]
    downsample_10hz = np.floor(downsample_10hz[:, 3 - 1 :: 3] / 3)
    return downsample_10hz


def _sum_counts(
    downsample_10hz: npt.NDArray[np.float_], epoch_seconds: int
) -> npt.NDArray[np.float64]:
    """Generate counts.

    Parameters
    ----------
    downsample_10hz:
        Matrix containing downsampled to 10hz data
    epoch_seconds:
        Used to compute how many raw samples are used for
        computing an epoch

    Returns
    -------
    epochs :
        The epochs
    """
    logger.info("Summing epochs")
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
    interpolate=False,
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
    interpolate:
        Use interpolation and mne low-pass filters for down-sampling to 30Hz instead of
        legacy filters.

    Returns
    -------
    epochs :
        The epochs
    """
    downsample_data = _resample(
        raw=raw, frequency=frequency, use_mne_filter=interpolate
    )
    del raw
    bpf_data = _bpf_filter(downsample_data=downsample_data)
    del downsample_data
    trim_data = _trim_data(bpf_data=bpf_data, lfe_select=lfe_select)
    del bpf_data
    downsample_10hz = _resample_10hz(trim_data=trim_data)
    del trim_data
    gc.collect(1)
    epoch_counts = _sum_counts(
        downsample_10hz=downsample_10hz, epoch_seconds=epoch_seconds
    )
    return epoch_counts


def get_counts(
    raw,
    freq: int,
    epoch: int,
    fast: bool = True,
    use_mne: bool = False,
):
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
    use_mne:
        Use MNE to directly down-sample data to 30Hz, instead of using legacy ActiGraph
        down-sampling method which maintains identical counts.

    Down-sampling from frequencies other than 30, 60 and 90 involves a low pass filter
    in order to avoid aliasing. The low-pass filter used in ActiGraph devices before
    Leap, as well as ActiLife and CenterPoint, needs to be fast and as thus is not
    immune to leaking higher frequencies [1]_.

    Returns
    -------
    counts : ndarray, shape (n_epochs, ANY)
        The counts, n_epochs = ceil(n_samples/freq).

    References
    ----------
    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    """
    if freq in [32, 64, 128, 256]:
        raw = resample_to_30hz(raw, freq, use_mne_filter=use_mne)  # type: ignore
        freq = 30
    else:
        assert use_mne or (freq in [30, 40, 50, 60, 70, 80, 90, 100]), (
            "freq must be in [30 : 10 : 100], or a power of 2 between 32 and 256."
            "Try setting `interpolate=True` otherwise."
        )

    if fast:
        counts = _extract(raw, freq, False, epoch, use_mne).transpose()
    else:
        axis_counts = []
        for i in range(raw.shape[1]):
            print(f"Running extract for axis {i} of {raw.shape[1]}")
            axis_counts.append(_extract_slow(raw[0:, [i]], freq, False, epoch))

        counts = np.concatenate(axis_counts, 0).transpose()

    return counts.astype(int)
