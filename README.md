# GaiaCMD

A Python package for plotting Gaia Color-Magnitude Diagrams for TESS white dwarf pulsation catalogs.

Give it a CSV of TIC IDs — it queries Gaia EDR3 and produces a CMD with all your targets plotted.

---

## Install

```bash
pip install gaiacmd
```

---

## Usage

### Python

```python
from gaiacmd import run_cmd, plot_cmd

state = run_cmd("my_tics.csv")
plot_cmd(state)
```

### Command line

```bash
gaiacmd --csv my_tics.csv
```

---

## Input format

A CSV file with a `TIC` column:

```
TIC
141872267
123456789
987654321
```

---

## Output

- A Gaia CMD figure with your WD targets plotted as red dots
- A grey background cloud of field stars (1° radius per target)
- The DAV (ZZ Ceti) instability strip overlay
- Saved as `gaia_cmd_wds.png`

---

## Resume support

Progress is saved to a checkpoint file after every TIC. If the run is interrupted, just re-run the same command — it will pick up where it left off. Delete `gaia_cmd_checkpoint.pkl` to start fresh.

---

## Options

| Argument | Default | Description |
|---|---|---|
| `--csv` | required | Path to your TIC list CSV |
| `--tic-column` | `TIC` | Column name for TIC IDs |
| `--output` | `gaia_cmd_wds.png` | Output figure filename |
| `--checkpoint` | `gaia_cmd_checkpoint.pkl` | Checkpoint file path |
| `--log` | `skipped_tics.log` | Skipped objects log path |
| `--radius` | `1.0` | Gaia cone search radius in degrees |

---

## Citation

If you use this package in your research, please cite the Gaia EDR3 catalog and the TESS mission.

---

## Author

[Nikoohn](https://github.com/Nikoohn)
