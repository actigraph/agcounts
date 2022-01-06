"""Miscellaneous data generation function."""
from pathlib import Path
from typing import List

import numpy as np
from numpy import typing as npt


def generate_data(epochs: List[int], frequencies: List[int], data_path: Path):
    """Randomly generate signal data.

    The data is saved to csv files.

    Parameters
    ----------
    epochs
        list of umber of epochs
    frequencies
        list of frequencies to generate
    """
    n_epochs = 1000
    mean = 0
    std = 1
    for epoch, freq in zip(epochs, frequencies):
        # Create white noise
        num_samples = int(n_epochs * freq * epoch)
        signal_x = np.random.normal(mean, std, size=num_samples)
        signal_y = np.random.normal(mean, std, size=num_samples)
        signal_z = np.random.normal(mean, std, size=num_samples)

        # Dump to CSV which can be opened in ActiLife
        signal: npt.NDArray[np.float_] = np.array([signal_x, signal_y, signal_z]).T
        np.savetxt(data_path / f"raw_{epoch}_{freq}.csv", signal, delimiter=",")
