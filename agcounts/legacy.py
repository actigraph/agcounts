"""Legacy functions in agcounts."""

import gc
import logging

import numpy as np
from numpy import typing as npt

logger = logging.getLogger(__name__)
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


def _extract_slow(
    raw: npt.NDArray[np.float_],
    frequency: int,
    lfe_select: bool,
    epoch: int,
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

    logger.info("Up-sampling data")
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
    logger.info("Down-sampling data")
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

    logger.info("Filtering data")
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

    logger.info("Threshold/trimming data")

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

    logger.info("Getting data back to 10Hz for accumulation")
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

    logger.info("Summing epochs")
    for i in range(len(epoch_counts[0])):
        epoch_counts[0, i] = np.floor(
            sum(down_sample10_hz[0, i * block_size : i * block_size + block_size])
        )
    del down_sample10_hz
    gc.collect(1)
    return epoch_counts


def _factors(frequency: int):
    factors = {
        30: (1, 1),
        40: (3, 4),
        50: (3, 5),
        60: (1, 2),
        70: (3, 7),
        80: (3, 8),
        90: (1, 3),
        100: (3, 10),
    }
    return factors.get(frequency, (1, 1))
