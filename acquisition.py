from typing import Union
import numpy as np
import uproot
import os
from pathlib import Path
from multiprocessing import Process
from tqdm import tqdm
import lecroyscope
from caenhv import CaenHV

from analysis import update_file_with_analysis, drift_times_analysis

print("Starting acquisition...")

# Run parameters
drift_gap = 5.0  # mm
quencher_pct = 2.0 # quencher percentage
# HV power supply parameters
mesh_voltage = 350.0  # V

drift_voltages = np.arange(20, 490, 10)
drift_voltages = [80]
# drift_voltages = [v for v in np.arange(110, 600, 5) if v not in drift_voltages]
print(f"Drift voltages: {drift_voltages}")

# Oscilloscope parameters
scope_ip = "192.168.10.5"
scope_n_sequences = 10
scope_num_segments = 500

# run directory
runs_dir = Path("/media/lo272082/Transcend/drift-velocity/data/05-12-2022")
runs = list(filter(os.path.isdir, map(lambda x: runs_dir / x, os.listdir(runs_dir)))) if runs_dir.exists() else []
run_numbers = set()
for run in runs:
    try:
        run_numbers.add(int(run.name.split("_")[1]))
    except ValueError:
        pass
    except IndexError:
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

def analysis(filename):
    update_file_with_analysis(filename)
    p = Path(filename)
    new_name = p.parents[0] / Path("analysis_" + p.name)
    os.rename(filename, new_name)

    with uproot.open(new_name) as f:
        tree = f["t"]
        for time_observable in ["RT20", "RT30", "RT40", "RT50", "RT90"]:
            times = tree[time_observable].array()
            mean, sigma = drift_times_analysis(times)
            print(f"{time_observable}: {mean} +- {sigma}")

analysis_process = None
try:
    hv_drift.on()
    hv_mesh.on()
    for voltage in drift_voltages:
        print(f"Mesh voltage: {mesh_voltage:0.1f} V, Drift voltage: {voltage:0.1f} V")
        root_filename = run_dir / Path(
            f"gap_{drift_gap:0.1f}_mesh_{mesh_voltage:0.1f}_drift_{voltage:0.1f}.root"
        )
        with uproot.recreate(root_filename) as output_file:

            hv_drift.vset = voltage
            hv_mesh.vset = mesh_voltage

            wait_vset_opt = {"timeout":20}
            hv_drift.wait_for_vset(**wait_vset_opt)
            hv_mesh.wait_for_vset(**wait_vset_opt)

            print("Target voltages reached")

            tree = None
            for i in tqdm(range(scope_n_sequences), desc=str(root_filename)):
                if not scope.acquire():
                    continue

                trace_group = scope.read(2, 3)

                if tree is None:
                    branches = lecroyscope.writing.root.get_tree_branch_definitions(trace_group, drift_voltage=float,
                                                                                    mesh_voltage=float, drift_gap=float,quencher_pct=float)
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
                    quencher_pct=quencher_pct,
                )
                tree.extend(data)

        if analysis_process is not None:
            analysis_process.join()
        analysis_process = Process(target=analysis, args=(root_filename,))
        analysis_process.start()

finally:
    hv_drift.off()
    hv_mesh.off()
    if analysis_process is not None:
        analysis_process.join()

print("Finished acquisition!")
