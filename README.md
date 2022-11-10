# Drift Velocity Measurements

## Requirements

All requirements are in `requirements.txt` which can be installed via `pip install -r requirements.txt`.

We use [this library](https://github.com/lobis/caen-hv) to interface with the CAEN HV power supply
and [this library](https://github.com/lobis/lecroy-scope) to interface with the LeCroy oscilloscope for data
acquisition.

```bash
pip install caenhv
pip install lecroyscope
```

We also use [`uproot`](https://github.com/scikit-hep/uproot5) to save the data directly into a root tree.

# Acquisition

```bash
python acquisition.py
```

# Analysis

