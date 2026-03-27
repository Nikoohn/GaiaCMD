"""
gaiacmd — Gaia CMD plotter for TESS white dwarf pulsation catalogs.

Usage
-----
    from gaiacmd import run_cmd, plot_cmd

    state = run_cmd("my_tics.csv")
    plot_cmd(state)
"""

from .cmd import run_cmd, plot_cmd

__version__ = "0.1.0"
__all__      = ["run_cmd", "plot_cmd"]
