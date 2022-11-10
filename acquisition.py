import time
import matplotlib.pyplot as plt
import numpy as np
import uproot
import awkward as ak
import os
from pathlib import Path
import lecroyscope

run = 21

scope = lecroyscope.Scope("192.168.10.5")

print(scope.id)

scope.sample_mode = "Sequence"
print(f"Sample mode: {scope.sample_mode}")

scope.num_segments = 200

from caenhv import CaenHV

caen = CaenHV()
module = caen[0]
hv_mesh = module.channel(1)
hv_drift = module.channel(2)

hv_mesh.rup = 20.0
hv_drift.rup = 20.0

run_dir = Path(f"run_{run:03d}")
os.mkdir(run_dir)

# run parameters
drift_gap = 10.0  # mm
mesh_voltage = 340.0  # V
hv_mesh.vset = mesh_voltage

n_sequences = 100

hv_drift.on()
hv_mesh.on()

# voltages = np.arange(312.5, 800, 50)
voltages = np.arange(95, 815, 15)


def _make_tree(file, _trace_group):
    _n_points = len(_trace_group.time)
    channels = {f"CH{trace.channel}": ("f4", (_n_points,)) for trace in _trace_group}
    return file.mktree(
        "t",
        {"time": ("f4", (_n_points,)), "drift_voltage": np.float32, "mesh_voltage": np.float32, "drift_gap": np.float32,
         **channels}, "Drift Velocity Measurement"
    )


# voltages = np.linspace(50, 850, 81)[14:]
# voltages = np.concatenate(([200], np.arange(650, 875, 25),))
print(f"Voltages: {voltages}")

for voltage in voltages:
    print(f"Mesh voltage: {mesh_voltage:0.1f} V, Drift voltage: {voltage:0.1f} V")

    root_filename = run_dir / Path(
        f"gap_{drift_gap:0.1f}_mesh_{mesh_voltage:0.1f}_drift_{voltage:0.1f}.root"
    )
    output_file = uproot.recreate(root_filename)

    hv_drift.vset = voltage
    hv_mesh.vset = mesh_voltage

    hv_drift.wait_for_vset(timeout=60)
    hv_mesh.wait_for_vset(timeout=60)

    n_points = None
    tree = None
    for i in range(n_sequences):
        print(f"sequence {i + 1} of {n_sequences}")
        if not scope.acquire(timeout=60):
            continue

        trace_group = scope.read(2, 3)

        if tree is None:
            tree = _make_tree(output_file, trace_group)

        n = len(trace_group[2])
        data = {f"CH{trace.channel}": trace.y for trace in trace_group}
        data_config = {"drift_voltage": n * [voltage], "mesh_voltage": n * [mesh_voltage], "drift_gap": n * [drift_gap]}
        data = {"time": n * [trace_group.time], **data, **data_config}
        tree.extend(data)

    output_file.close()
