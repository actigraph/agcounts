from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agcounts.extract import get_counts


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    if not config.getoption("--runslow"):
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


data_path = Path("./data")
raw_data_path = data_path / "raw"
cp_data_path = data_path / "CentrePointCounts"
al_data_path = data_path / "ActiLifeCounts"


@pytest.fixture
def agcounts_serial(request):
    epoch, freq = request.param
    signals = pd.read_csv(
        raw_data_path / f"raw_{epoch}_{freq}.csv.gz", skiprows=0, header=None
    )
    counts = get_counts(np.array(signals), freq=freq, epoch=epoch, fast=False)
    return counts


@pytest.fixture
def agcounts(request):
    epoch, freq = request.param
    signals = pd.read_csv(
        raw_data_path / f"raw_{epoch}_{freq}.csv.gz", skiprows=0, header=None
    )
    counts = get_counts(signals.values, freq=freq, epoch=epoch, fast=True)
    return counts


@pytest.fixture
def noisy_agcounts(request):
    epoch, freq, offset = request.param
    signals = pd.read_csv(
        raw_data_path / f"raw_{epoch}_{freq}.csv.gz", skiprows=0, header=None
    ).values
    signals[:, 2] = signals[:, 2] + offset
    counts = get_counts(signals, freq=freq, epoch=epoch, fast=True)
    return counts


@pytest.fixture
def agcounts_better_lpf(request):
    epoch, freq = request.param
    signals = pd.read_csv(
        raw_data_path / f"raw_{epoch}_{freq}.csv.gz", skiprows=0, header=None
    )
    counts = get_counts(
        np.array(signals), freq=freq, epoch=epoch, fast=True, use_mne=True
    )
    return counts


@pytest.fixture
def cp_counts(request):
    epoch, freq = request.param
    counts = pd.read_csv(
        cp_data_path / f"raw_{epoch}_{freq}_CentrePointCounts.csv", skiprows=0
    )
    counts_axes = counts[["X (axis 2)", "Y (axis 1)", "Z (axis3)"]].to_numpy()
    return counts_axes


@pytest.fixture
def al_counts(request):
    epoch, freq = request.param
    counts = pd.read_csv(
        al_data_path / f"raw_{epoch}_{freq}_counts{epoch}sec.csv", skiprows=10
    )
    counts_axes = counts[["Axis2", "Axis1", "Axis3"]].to_numpy()
    return counts_axes


@pytest.fixture
def input_32hz():
    df = pd.read_csv(data_path / "32Hz_input.csv", header=None)
    return df.values


@pytest.fixture
def upsample_256():
    df = pd.read_csv(data_path / "upsample256.csv", header=None)
    return df.values


@pytest.fixture
def upsample_256_lpf():
    df = pd.read_csv(data_path / "upsample256Lpf.csv", header=None)
    return df.values


@pytest.fixture
def taso_256_lpf():
    df = pd.read_csv(data_path / "taso256HzLpf.csv", header=None)
    return df.values


@pytest.fixture
def output_30hz():
    df = pd.read_csv(data_path / "30Hz_output.csv", header=None)
    return df.values
