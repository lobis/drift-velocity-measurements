import time
import matplotlib.pyplot as plt
import numpy as np
import uproot
import awkward as ak
import os
from pathlib import Path
import lecroyscope

scope = lecroyscope.Scope("192.168.10.5")

run = 4
n_sequences = 8
scope.num_segments = 2500
scope.timeout = 10 * 60.

print(scope.id)

scope.trigger_mode = "Stopped"
scope.sample_mode = "Sequence"
print(f"Sample mode: {scope.sample_mode}")

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

hv_drift.on()
hv_mesh.on()

# voltages = list(reversed(np.arange(40, 1015, 15)))
voltages = [145., 160.]

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

    tree = None
    for i in range(n_sequences):
        print(f"sequence {i + 1} of {n_sequences}")
        if not scope.acquire():
            continue

        trace_group = scope.read(2, 3)

        if tree is None:
            branches = lecroyscope.writing.root.get_tree_branch_definitions(trace_group, drift_voltage=float,
                                                                            mesh_voltage=float, drift_gap=float)
            tree = output_file.mktree(
                "t",
                branches,
                "EventTree",
            )

        data = lecroyscope.writing.root.get_tree_extend_data(
            trace_group,
            drift_voltage=voltage,
            mesh_voltage=mesh_voltage,
            drift_gap=drift_gap,
        )
        tree.extend(data)

    output_file.close()

hv_drift.off()
hv_mesh.off()
