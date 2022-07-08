"""Functions for dealing with pow2 sample rates."""
import math

import numpy as np


def resample_to_30hz(data, sample_rate):
    """Resample data to 30Hz."""
    data = upsample_to_256hz(data, sample_rate)
    data = taso_lpf(data)
    data = interpolated_resample(data)

    return data


def upsample_to_256hz(data, sample_rate):
    """Upsample data to 256Hz."""
    if sample_rate == 256:
        return data
    else:
        assert sample_rate in [32, 64, 128]
        factor = 256 // sample_rate
    m, n = data.shape
    out = np.zeros((m * factor, n), dtype=data.dtype)
    out[::factor, :] = data * factor
    # LPF
    ker = np.array(
        [
            -0.000001,
            -0.000002,
            -0.000004,
            -0.000005,
            -0.000006,
            -0.000006,
            -0.000004,
            0.000000,
            0.000005,
            0.000011,
            0.000017,
            0.000022,
            0.000023,
            0.000020,
            0.000013,
            -0.000000,
            -0.000016,
            -0.000033,
            -0.000049,
            -0.000059,
            -0.000061,
            -0.000052,
            -0.000031,
            -0.000000,
            0.000038,
            0.000077,
            0.000111,
            0.000132,
            0.000133,
            0.000112,
            0.000066,
            -0.000000,
            -0.000078,
            -0.000156,
            -0.000222,
            -0.000260,
            -0.000259,
            -0.000214,
            -0.000125,
            0.000000,
            0.000145,
            0.000288,
            0.000404,
            0.000469,
            0.000464,
            0.000380,
            0.000220,
            -0.000000,
            -0.000250,
            -0.000493,
            -0.000687,
            -0.000791,
            -0.000777,
            -0.000632,
            -0.000363,
            0.000000,
            0.000409,
            0.000801,
            0.001108,
            0.001269,
            0.001240,
            0.001003,
            0.000574,
            -0.000000,
            -0.000639,
            -0.001246,
            -0.001715,
            -0.001956,
            -0.001903,
            -0.001534,
            -0.000873,
            0.000000,
            0.000966,
            0.001875,
            0.002574,
            0.002926,
            0.002839,
            0.002280,
            0.001295,
            -0.000000,
            -0.001425,
            -0.002762,
            -0.003783,
            -0.004293,
            -0.004158,
            -0.003335,
            -0.001892,
            0.000000,
            0.002078,
            0.004025,
            0.005513,
            0.006257,
            0.006061,
            0.004866,
            0.002763,
            -0.000000,
            -0.003045,
            -0.005909,
            -0.008115,
            -0.009237,
            -0.008981,
            -0.007240,
            -0.004131,
            0.000000,
            0.004606,
            0.009005,
            0.012468,
            0.014325,
            0.014075,
            0.011483,
            0.006641,
            -0.000000,
            -0.007655,
            -0.015271,
            -0.021640,
            -0.025543,
            -0.025900,
            -0.021926,
            -0.013250,
            0.000000,
            0.017180,
            0.037161,
            0.058432,
            0.079245,
            0.097796,
            0.112422,
            0.121780,
            0.125000,
            0.121780,
            0.112422,
            0.097796,
            0.079245,
            0.058432,
            0.037161,
            0.017180,
            0.000000,
            -0.013250,
            -0.021926,
            -0.025900,
            -0.025543,
            -0.021640,
            -0.015271,
            -0.007655,
            -0.000000,
            0.006641,
            0.011483,
            0.014075,
            0.014325,
            0.012468,
            0.009005,
            0.004606,
            0.000000,
            -0.004131,
            -0.007240,
            -0.008981,
            -0.009237,
            -0.008115,
            -0.005909,
            -0.003045,
            -0.000000,
            0.002763,
            0.004866,
            0.006061,
            0.006257,
            0.005513,
            0.004025,
            0.002078,
            0.000000,
            -0.001892,
            -0.003335,
            -0.004158,
            -0.004293,
            -0.003783,
            -0.002762,
            -0.001425,
            -0.000000,
            0.001295,
            0.002280,
            0.002839,
            0.002926,
            0.002574,
            0.001875,
            0.000966,
            0.000000,
            -0.000873,
            -0.001534,
            -0.001903,
            -0.001956,
            -0.001715,
            -0.001246,
            -0.000639,
            -0.000000,
            0.000574,
            0.001003,
            0.001240,
            0.001269,
            0.001108,
            0.000801,
            0.000409,
            0.000000,
            -0.000363,
            -0.000632,
            -0.000777,
            -0.000791,
            -0.000687,
            -0.000493,
            -0.000250,
            -0.000000,
            0.000220,
            0.000380,
            0.000464,
            0.000469,
            0.000404,
            0.000288,
            0.000145,
            0.000000,
            -0.000125,
            -0.000214,
            -0.000259,
            -0.000260,
            -0.000222,
            -0.000156,
            -0.000078,
            -0.000000,
            0.000066,
            0.000112,
            0.000133,
            0.000132,
            0.000111,
            0.000077,
            0.000038,
            -0.000000,
            -0.000031,
            -0.000052,
            -0.000061,
            -0.000059,
            -0.000049,
            -0.000033,
            -0.000016,
            -0.000000,
            0.000013,
            0.000020,
            0.000023,
            0.000022,
            0.000017,
            0.000011,
            0.000005,
            0.000000,
            -0.000004,
            -0.000006,
            -0.000006,
            -0.000005,
            -0.000004,
            -0.000002,
            -0.000001,
        ]
    )
    for i in range(n):
        out[:, i] = np.convolve(out[:, i], ker, "same")
    return out


def taso_lpf(data):
    """Now at 256 Hz, we apply a single pole lowpass IIR filter."""
    b = 0.307990357416655
    a = 0.692009642583345

    out = np.zeros_like(data)
    y = data[0, :] * 0
    x = b * data

    # Transient step response.
    for i in range(180):
        y = x[0, :] + a * y

    # Filtering
    for i in range(data.shape[0]):
        y = x[i, :] + a * y
        out[i, :] = y

    return out


def interpolated_resample(data):
    """Then we do a linear interpolation to get 256 Hz to 30 Hz."""
    m, n = data.shape
    out = np.zeros((data.shape[0] * 30 // 256, n))

    # Resample
    for i in range(1, out.shape[0]):
        # Compute 2 nearest neighbors in original data stream.
        rename = math.floor((256 / 30) * i) - 1

        # Linear regression.
        a = data[rename + 1, :] - data[rename, :]
        b = data[rename, :] - a * (rename + 1)

        # Compute interpolation.
        out[i - 1, :] = ((256 / 30) * i) * a + b
    return out
