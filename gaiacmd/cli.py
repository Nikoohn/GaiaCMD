"""
gaiacmd/cli.py
--------------
Command-line interface for gaiacmd.

Usage:
    gaiacmd --csv my_tics.csv
    gaiacmd --csv my_tics.csv --output my_cmd.png --radius 1.5
"""

import argparse
from .cmd import run_cmd, plot_cmd


def main():
    parser = argparse.ArgumentParser(
        prog="gaiacmd",
        description="Plot a Gaia CMD for a catalog of TESS white dwarf targets.",
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to your TIC list CSV file (must have a TIC column).",
    )
    parser.add_argument(
        "--tic-column", default="TIC",
        help="Column name for TIC IDs in the CSV (default: TIC).",
    )
    parser.add_argument(
        "--output", default="gaia_cmd_wds.png",
        help="Output figure filename (default: gaia_cmd_wds.png).",
    )
    parser.add_argument(
        "--checkpoint", default="gaia_cmd_checkpoint.pkl",
        help="Checkpoint file for resume support (default: gaia_cmd_checkpoint.pkl).",
    )
    parser.add_argument(
        "--log", default="skipped_tics.log",
        help="Log file for skipped objects (default: skipped_tics.log).",
    )
    parser.add_argument(
        "--radius", type=float, default=1.0,
        help="Gaia cone search radius in degrees (default: 1.0).",
    )

    args = parser.parse_args()

    state = run_cmd(
        csv_file         = args.csv,
        tic_column       = args.tic_column,
        log_file         = args.log,
        checkpoint_file  = args.checkpoint,
        field_radius_deg = args.radius,
    )

    plot_cmd(
        state,
        output_plot = args.output,
    )


if __name__ == "__main__":
    main()
