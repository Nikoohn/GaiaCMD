"""
gaiacmd — Gaia CMD plotter for TESS white dwarf pulsation catalogs.
 
Usage (Python)
--------------
    from gaiacmd import run_cmd, plot_cmd
 
    state = run_cmd("my_tics.csv")
    plot_cmd(state)
 
Usage (CLI)
-----------
    gaiacmd --csv my_tics.csv
"""
 
from .cmd import run_cmd, plot_cmd
from .cli import main
 
__version__ = "0.2.0"
__all__      = ["run_cmd", "plot_cmd", "main"]
 
