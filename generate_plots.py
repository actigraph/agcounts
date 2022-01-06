"""Check epoch lengths [10, 30] and all sampling frequencies.

 In order to obtain ActiLife counts, copy over the title info from an
 ordinary csv raw file into the saved csv files, adjust the sampling
 frequency and open with ActiLife and convert to counts using
 raw->epochs functionality.

 60 and 90 Hz output from ActiLife do not make sense. They are treated
 just like 30 Hz."""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from agcounts.extract import get_counts

data_path = Path("./data")
raw_data_path = data_path / "raw"
cp_data_path = data_path / "CentrePointCounts"
al_data_path = data_path / "ActiLifeCounts"

FREQUENCIES = [30, 40, 50, 60, 70, 80, 90, 100]
EPOCHS = [30, 10]


def generate_plots(epochs: List[int], frequencies: List[int]):
    font = {"size": 16}
    plt.rc("font", **font)
    fig_size = (20, 20)
    plt.ioff()
    for i in range(3):
        _ = plt.figure(figsize=fig_size, dpi=300)
    all_errors = []
    counter = 0
    for epoch in tqdm(epochs):
        for freq in tqdm(frequencies):
            counter += 1
            counts_cp = pd.read_csv(
                cp_data_path / f"raw_{epoch}_{freq}_CentrePointCounts.csv", skiprows=0
            )
            counts_cp = counts_cp[["X (axis 2)", "Y (axis 1)", "Z (axis3)"]].to_numpy()

            # back for consistency
            signals = pd.read_csv(
                raw_data_path / f"raw_{epoch}_{freq}.csv", skiprows=0, header=None
            )
            counts_py = get_counts(np.array(signals), freq=freq, epoch=epoch)
            counts_py = counts_py.flatten()[6:]
            counts_cp = counts_cp.flatten()[6:]

            min_length = np.min([counts_py.shape, counts_cp.shape])
            errors = counts_py[:min_length] - counts_cp[:min_length]
            all_errors.append(errors)
            mean = (counts_py + counts_cp) / 2
            mean_abs_rel_errors = np.abs(errors / counts_cp * 100)
            np.mean(mean_abs_rel_errors)
            np.std(mean_abs_rel_errors)

            # Bland-altman plot
            plt.figure(1)
            plt.subplot(len(epochs) * 2, len(frequencies) // 2, counter)
            plt.gca().title.set_text(f"f={freq}Hz, e={epoch}")
            plt.scatter(mean, errors, s=2, color="k")
            plt.xlabel("Counts (mean)")
            plt.ylabel("Error (counts)")
            plt.axhline(y=np.mean(errors), color="red", linestyle="--", linewidth=1)
            plt.axhline(
                y=np.mean(errors) + np.std(errors) * 1.96,
                color="orange",
                linestyle="--",
                linewidth=1,
            )
            plt.axhline(
                y=np.mean(errors) - np.std(errors) * 1.96,
                color="orange",
                linestyle="--",
                linewidth=1,
            )

            # Histogram of errors
            plt.figure(2)
            plt.subplot(len(epochs) * 2, len(frequencies) // 2, counter)
            plt.gca().title.set_text(f"f={freq}Hz, e={epoch}")
            plt.hist(errors, bins=70, color="gray")
            plt.ylabel("Number of epochs")
            plt.xlabel("Error (counts)")

            # Histogram of mean absolute percentage error
            plt.figure(3)
            plt.subplot(len(epochs) * 2, len(frequencies) // 2, counter)
            plt.gca().title.set_text(f"f={freq}Hz, e={epoch}")
            plt.hist(mean_abs_rel_errors, color="gray")
            plt.ylabel("Number of epochs")
            plt.xlabel("Absolute percentage error")

    for num, name in {1: "bland_altman", 2: "hist", 3: "epochs"}.items():
        plt.figure(num)
        plt.tight_layout()
        plt.savefig(f"figs/{name}.pdf")
        plt.savefig(f"figs/{name}.png")
        plt.close()


if __name__ == "__main__":
    generate_plots([30, 10], [30, 40])
