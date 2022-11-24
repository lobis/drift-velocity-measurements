from typing import Union
import numpy as np
import uproot
import os
from pathlib import Path

import lecroyscope
from caenhv import CaenHV

# Run parameters
drift_gap = 10.0  # mm

# HV power supply parameters
mesh_voltage = 340.0  # V
drift_voltages = np.arange(95, 815, 15)
print(f"Drift voltages: {drift_voltages}")

# Oscilloscope parameters
scope_ip = "192.168.10.5"
scope_n_sequences = 8
scope_num_segments = 2500

# run directory
runs_dir = Path("runs")
runs = list(filter(os.path.isdir, map(lambda x: runs_dir / x, os.listdir(runs_dir)))) if runs_dir.exists() else []
run_numbers = set()
for run in runs:
    try:
        run_numbers.add(int(run.name.split("_")[1]))
    except Union[ValueError, IndexError]:
        pass

run_number = max(run_numbers) + 1 if len(run_numbers) > 0 else 1
run_dir = runs_dir / f"run_{run_number:04d}"
run_dir.mkdir(parents=True, exist_ok=True)

# Set instruments
scope = lecroyscope.Scope(scope_ip)
scope.trigger_mode = "Stopped"
scope.sample_mode = "Sequence"
scope.timeout = 10 * 60.
scope.num_segments = scope_num_segments
print(f"Scope ID: {scope.id} - OK")

caen = CaenHV()
module = caen[0]
hv_mesh = module.channel(1)
hv_drift = module.channel(2)

hv_mesh.rup = 20.0
hv_drift.rup = 20.0

hv_mesh.vset = mesh_voltage

print("CAEN HV power supply - OK")

try:
    hv_drift.on()
    hv_mesh.on()

    for voltage in drift_voltages:
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
        for i in range(scope_n_sequences):
            print(f"sequence {i + 1} of {scope_n_sequences}")
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

finally:
    hv_drift.off()
    hv_mesh.off()
