"""
gaiacmd/cmd.py
--------------
Core logic for querying Gaia EDR3 and plotting a Color-Magnitude Diagram
for a catalog of TESS white dwarf targets.
"""

import pickle
import warnings
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia

warnings.filterwarnings("ignore", category=UserWarning, append=True)

# ----------------------------------------------------------------------
# Default constants (can be overridden via run_cmd() arguments)
# ----------------------------------------------------------------------
DEFAULT_FIELD_RADIUS_DEG = 1.0
DEFAULT_MAX_MATCH_ARCSEC = 10.0
DEFAULT_GAIA_TABLE       = "gaiaedr3.gaia_source"
DEFAULT_MAX_RETRIES      = 3
DEFAULT_RETRY_DELAY      = 10   # seconds between retries on failure
DEFAULT_QUERY_SPACING    = 2    # seconds between every successful query

# Approximate DAV (ZZ Ceti) instability strip corners [BP-RP, M_G]
DAV_COLOR = [0.00, 0.00, 0.25, 0.20]
DAV_MG    = [11.8, 12.8, 13.2, 12.2]


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _safe_float_col(table, col):
    """Return a numpy float64 array from an astropy table column."""
    return np.array(table[col], dtype=np.float64)


def _compute_cmd(g, bp, rp, plx):
    """Return (BP-RP color, absolute G magnitude)."""
    M_G   = g + 5.0 * np.log10(plx) - 10.0
    color = bp - rp
    return color, M_G


def _gaia_field_query(ra, dec, radius_deg,
                      gaia_table, max_retries, retry_delay, query_spacing):
    """
    Query Gaia EDR3 for all well-measured sources within radius_deg of
    (ra, dec). Returns an astropy Table or raises on persistent failure.
    """
    adql = f"""
        SELECT source_id, ra, dec,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, parallax
        FROM   {gaia_table}
        WHERE  1=CONTAINS(
                   POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
               )
        AND    parallax          >  0
        AND    phot_g_mean_mag   IS NOT NULL
        AND    phot_bp_mean_mag  IS NOT NULL
        AND    phot_rp_mean_mag  IS NOT NULL
    """
    for attempt in range(1, max_retries + 1):
        try:
            job    = Gaia.launch_job_async(adql)
            result = job.get_results()
            time.sleep(query_spacing)
            return result
        except Exception as exc:
            if attempt < max_retries:
                print(f"    Gaia query attempt {attempt} failed ({exc}). "
                      f"Retrying in {retry_delay}s ...")
                time.sleep(retry_delay)
            else:
                raise


def _load_checkpoint(path):
    """Load saved state from disk, or return a clean initial state."""
    p = Path(path)
    if p.exists():
        with p.open("rb") as fh:
            state = pickle.load(fh)
        n_done = len(state["done_tics"])
        n_wds  = len(state["wd_colors"])
        print(f"  Checkpoint found: {n_done} TIC(s) already processed, "
              f"{n_wds} WD(s) collected.  Resuming ...\n")
        return state
    print("  No checkpoint found — starting fresh.\n")
    return {
        "done_tics":       set(),
        "wd_colors":       [],
        "wd_mgs":          [],
        "wd_labels":       [],
        "field_colors":    [],
        "field_mgs":       [],
        "seen_source_ids": set(),
        "skipped":         [],
    }


def _save_checkpoint(path, state):
    """Atomically write state to disk (write tmp then rename)."""
    tmp = Path(path).with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _append_skip_log(path, entries):
    """Append skipped entries to the log file (creates header if new)."""
    if not entries:
        return
    p = Path(path)
    write_header = not p.exists()
    with p.open("a") as fh:
        if write_header:
            fh.write("TIC_ID,Reason\n")
        for tid, reason in entries:
            fh.write(f"{tid},{reason}\n")


# ======================================================================
# PUBLIC API
# ======================================================================

def run_cmd(
    csv_file,
    tic_column          = "TIC",
    log_file            = "skipped_tics.log",
    checkpoint_file     = "gaia_cmd_checkpoint.pkl",
    field_radius_deg    = DEFAULT_FIELD_RADIUS_DEG,
    max_match_arcsec    = DEFAULT_MAX_MATCH_ARCSEC,
    gaia_table          = DEFAULT_GAIA_TABLE,
    max_retries         = DEFAULT_MAX_RETRIES,
    retry_delay         = DEFAULT_RETRY_DELAY,
    query_spacing       = DEFAULT_QUERY_SPACING,
):
    """
    Query Gaia for every TIC in csv_file and collect CMD data.

    Parameters
    ----------
    csv_file        : str  — path to your CSV (must have a TIC ID column)
    tic_column      : str  — name of the TIC ID column (default: "TIC")
    log_file        : str  — path for the skipped-objects log
    checkpoint_file : str  — path for the resume checkpoint (delete to restart)
    field_radius_deg: float — Gaia cone search radius in degrees
    max_match_arcsec: float — max TIC-Gaia separation to accept a match
    gaia_table      : str  — Gaia table to query
    max_retries     : int  — number of retry attempts on Gaia failure
    retry_delay     : int  — seconds to wait between retries
    query_spacing   : int  — seconds to wait between every query

    Returns
    -------
    state : dict with keys wd_colors, wd_mgs, wd_labels,
                           field_colors, field_mgs, skipped
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- load CSV ---
    df = pd.read_csv(csv_file)
    if tic_column not in df.columns:
        raise ValueError(f"Column '{tic_column}' not found in {csv_file}. "
                         f"Available columns: {list(df.columns)}")
    tic_ids = df[tic_column].astype(int).tolist()
    n_total = len(tic_ids)
    print(f"Loaded {n_total} TIC IDs from '{csv_file}'.")

    # --- load checkpoint ---
    state           = _load_checkpoint(checkpoint_file)
    done_tics       = state["done_tics"]
    wd_colors       = state["wd_colors"]
    wd_mgs          = state["wd_mgs"]
    wd_labels       = state["wd_labels"]
    field_colors    = state["field_colors"]
    field_mgs       = state["field_mgs"]
    seen_source_ids = state["seen_source_ids"]

    remaining   = [t for t in tic_ids if t not in done_tics]
    n_remaining = len(remaining)
    n_pad       = len(str(n_remaining))
    print(f"  {n_total - n_remaining} already done, {n_remaining} remaining.\n")

    new_skips = []

    try:
        for i, tic_id in enumerate(remaining):
            target_name = f"TIC {tic_id}"
            print(f"[{i+1:>{n_pad}}/{n_remaining}]  {target_name}",
                  end="  ...  ", flush=True)

            # 1. TIC catalog lookup
            try:
                tic_res = Catalogs.query_object(
                    target_name, catalog="TIC", radius=0.003 * u.deg
                )
                exact      = tic_res[tic_res["ID"] == str(tic_id)]
                row        = exact[0] if len(exact) > 0 else tic_res[0]
                target_ra  = float(row["ra"])
                target_dec = float(row["dec"])
            except Exception as exc:
                reason = f"TIC catalog lookup failed: {exc}"
                print(f"SKIPPED  ({reason})")
                new_skips.append((tic_id, reason))
                done_tics.add(tic_id)
                _save_checkpoint(checkpoint_file, state)
                continue

            # 2. Gaia field query
            try:
                field = _gaia_field_query(
                    target_ra, target_dec, field_radius_deg,
                    gaia_table, max_retries, retry_delay, query_spacing,
                )
            except Exception as exc:
                reason = f"Gaia query failed: {exc}"
                print(f"SKIPPED  ({reason})")
                new_skips.append((tic_id, reason))
                done_tics.add(tic_id)
                _save_checkpoint(checkpoint_file, state)
                continue

            if len(field) == 0:
                reason = "No Gaia sources in field"
                print(f"SKIPPED  ({reason})")
                new_skips.append((tic_id, reason))
                done_tics.add(tic_id)
                _save_checkpoint(checkpoint_file, state)
                continue

            # 3. Match closest Gaia source -> WD
            target_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
            field_coords = SkyCoord(ra=_safe_float_col(field, "ra")  * u.deg,
                                    dec=_safe_float_col(field, "dec") * u.deg)
            seps     = target_coord.separation(field_coords).arcsec
            best_idx = int(np.argmin(seps))
            best_sep = seps[best_idx]

            if best_sep > max_match_arcsec:
                reason = (f"Closest Gaia source is {best_sep:.1f}\" away "
                          f"(limit {max_match_arcsec}\")")
                print(f"SKIPPED  ({reason})")
                new_skips.append((tic_id, reason))
                done_tics.add(tic_id)
                _save_checkpoint(checkpoint_file, state)
                continue

            wd_row = field[best_idx]
            wd_plx = float(wd_row["parallax"])

            if wd_plx <= 0:
                reason = "Gaia match has non-positive parallax"
                print(f"SKIPPED  ({reason})")
                new_skips.append((tic_id, reason))
                done_tics.add(tic_id)
                _save_checkpoint(checkpoint_file, state)
                continue

            # 4. Accumulate WD CMD point
            wd_c, wd_m = _compute_cmd(
                float(wd_row["phot_g_mean_mag"]),
                float(wd_row["phot_bp_mean_mag"]),
                float(wd_row["phot_rp_mean_mag"]),
                wd_plx,
            )
            wd_colors.append(wd_c)
            wd_mgs.append(wd_m)
            wd_labels.append(tic_id)

            # 5. Accumulate deduplicated field stars
            g_arr   = _safe_float_col(field, "phot_g_mean_mag")
            bp_arr  = _safe_float_col(field, "phot_bp_mean_mag")
            rp_arr  = _safe_float_col(field, "phot_rp_mean_mag")
            plx_arr = _safe_float_col(field, "parallax")
            sid_arr = np.array(field["source_id"], dtype=np.int64)

            for j in range(len(field)):
                if j == best_idx:
                    continue
                sid = int(sid_arr[j])
                if sid in seen_source_ids:
                    continue
                seen_source_ids.add(sid)
                fc, fm = _compute_cmd(g_arr[j], bp_arr[j], rp_arr[j], plx_arr[j])
                field_colors.append(fc)
                field_mgs.append(fm)

            # 6. Save checkpoint
            done_tics.add(tic_id)
            _save_checkpoint(checkpoint_file, state)
            print(f"OK  (sep={best_sep:.2f}\", M_G={wd_m:.2f}, BP-RP={wd_c:.2f})")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.  All progress saved — re-run to continue.\n")
        _append_skip_log(log_file, new_skips)
        return state

    # --- write skip log ---
    _append_skip_log(log_file, new_skips)
    if new_skips:
        print(f"\n{len(new_skips)} object(s) skipped this run — appended to '{log_file}'")
    else:
        print("\nNo objects skipped this run.")
    print(f"{len(wd_colors)} WD(s) total collected for the CMD.\n")

    return state


def plot_cmd(
    state,
    output_plot = "gaia_cmd_wds.png",
    title       = "Gaia CMD — TESS WD Pulsation Catalog",
    xlim        = (-1.0, 3.0),
    ylim        = (18.0, 0.0),
):
    """
    Plot the Gaia CMD from the state dict returned by run_cmd().

    Parameters
    ----------
    state       : dict — returned by run_cmd()
    output_plot : str  — filename to save the figure (set to None to skip saving)
    title       : str  — plot title
    xlim        : tuple — (xmin, xmax) for BP-RP axis
    ylim        : tuple — (ymax, ymin) — note: y-axis is inverted (bright at top)
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    wd_colors    = state["wd_colors"]
    wd_mgs       = state["wd_mgs"]
    field_colors = state["field_colors"]
    field_mgs    = state["field_mgs"]

    fig, ax = plt.subplots(figsize=(9, 10))

    # grey field-star cloud
    if field_colors:
        ax.scatter(
            field_colors, field_mgs,
            s=1.5, c="grey", alpha=0.15, linewidths=0,
            rasterized=True,
            label=f"Field stars (1°/target, N={len(field_colors):,})",
            zorder=1,
        )

    # WD targets
    if wd_colors:
        ax.scatter(
            wd_colors, wd_mgs,
            s=35, c="crimson", alpha=0.85,
            edgecolors="darkred", linewidths=0.4,
            label=f"WD targets (N={len(wd_colors)})",
            zorder=5,
        )

    # DAV instability strip
    ax.fill(DAV_COLOR, DAV_MG, color="gold", alpha=0.22, zorder=3,
            label="DAV instability strip")
    ax.plot(DAV_COLOR + [DAV_COLOR[0]], DAV_MG + [DAV_MG[0]],
            "--", color="goldenrod", linewidth=1.3, zorder=4)

    ax.set_xlabel(r"$G_{\rm BP} - G_{\rm RP}$  [mag]", fontsize=13)
    ax.set_ylabel(r"$M_G$  [mag]", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    if output_plot:
        fig.savefig(output_plot, dpi=150, bbox_inches="tight")
        print(f"Plot saved to '{output_plot}'")
    plt.show()
    return fig, ax
