import numpy as np
import pytest

from agcounts.pow2 import (
    interpolated_resample,
    resample_to_30hz,
    taso_lpf,
    upsample_to_256hz,
)


def test_upsample_256_lpf(input_32hz, upsample_256_lpf):
    upsampled = upsample_to_256hz(input_32hz, 32)
    assert np.abs(upsampled - upsample_256_lpf).max() == pytest.approx(0)


def test_taso_256_lpf(upsample_256_lpf, taso_256_lpf):
    filtered = taso_lpf(upsample_256_lpf)
    assert np.abs(filtered - taso_256_lpf).max() == pytest.approx(0)


def test_interpolated(taso_256_lpf, output_30hz):
    interpolated = interpolated_resample(taso_256_lpf)
    assert np.abs(interpolated - output_30hz).max() == pytest.approx(0)


def test_32hz(input_32hz, output_30hz):
    resampled = resample_to_30hz(input_32hz, sample_rate=32)
    assert np.abs(resampled - output_30hz).max() == pytest.approx(0)


def test_32hz_interpolate(input_32hz, output_30hz):
    resampled = resample_to_30hz(input_32hz, sample_rate=32, interpolate=False)
    assert np.abs(resampled - output_30hz).max() == pytest.approx(0)
