import matplotlib.pyplot as plt
import numpy as np
import uproot
import scipy
import os
from pathlib import Path
from tqdm import tqdm


def peak_analysis(
        x: np.ndarray, peak_pcts: np.ndarray = np.array([0.5]), n_average: int = 50
):
    x = scipy.ndimage.uniform_filter1d(x, size=n_average)  # reduce high frequency noise

    peaks, _ = scipy.signal.find_peaks(x, distance=len(x) // 10)
    prominences = scipy.signal.peak_prominences(x, peaks)[0]

    # get the highest peak
    argmax = np.argmax(prominences)
    peak_index = peaks[argmax]
    peak_height = prominences[argmax]

    # compute relative peak height (prominence) and base
    peak_base = x[peak_index] - peak_height

    idxs = np.zeros(len(peak_pcts), dtype=int)
    for i, peak_pct in enumerate(peak_pcts):
        h_line = np.ones(len(x)) * (peak_base + peak_height * peak_pct)
        idxs[i] = sorted(np.argwhere(np.diff(np.sign(x - h_line))).flatten())[0]

    return idxs, peak_index, peak_base, peak_height


def update_file_with_analysis(filename):
    tree_name = "t"
    data = dict()
    with uproot.open(filename) as f:
        tree = f[tree_name]
        for key in tree.keys():
            data[key] = tree[key].array()

    peak_data = ["peak_position", "peak_height", "peak_base"]
    rise_times = np.arange(0.05, 1.0, 0.05)
    rise_time_to_string = lambda x: f"RT{int(x * 100):02d}"
    peak_data = [*peak_data, *[rise_time_to_string(x) for x in rise_times]]

    times = np.zeros(len(data["time"]))
    n = len(times)

    peak_data = {key: np.zeros(n) for key in peak_data}

    for i in tqdm(range(n)):
        if i % 1000 == 0:
            pass
            # print(f"{100*i/n:0.2f}%")
        x = data["time"][i]
        y = data["CH3"][i]
        idxs, peak_index, peak_base, peak_height = peak_analysis(
            y, peak_pcts=np.array(rise_times)
        )
        peak_data["peak_position"][i] = x[peak_index]
        peak_data["peak_height"][i] = peak_height
        peak_data["peak_base"][i] = peak_base
        for rise_time_i, rise_time in enumerate(rise_times):
            peak_data[rise_time_to_string(rise_time)][i] = x[idxs[rise_time_i]]

    data_new = {**data, **peak_data}
    for key in data:
        np.testing.assert_array_equal(data[key], data_new[key])
    for key in data_new:
        assert len(data_new[key]) == n

    with uproot.recreate(filename) as f:
        f[tree_name] = data_new
