"""Simple runner to plot Anton-Paar rheology text files.

Edit the `txt_path` variable below to point to your .txt export,
then run this script with a regular Python interpreter (e.g. F5 in an IDE).
This will save two PNG plots in the same folder as the text file:
    • <file>_temperature_ramp.png
    • <file>_flow_sweep.png
"""

import re
import io
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ------------ EDIT HERE ----------------------------------------------------
# Absolute or relative path to the exported .txt file you want to plot
# Example:
# txt_path = r"Rheology/daya_rheology/LLM_EP0901_0%01_Oscillatory temperature sweep.txt"
# ---------------------------------------------------------------------------
txt_path = 'Rheology/daya_rheology/LLM_EP0905_20%02_Oscillatory temperature sweep_Flow.txt'
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#                       Internal helper functions
# ---------------------------------------------------------------------------

STEP_HEADER_RE = re.compile(r"\[step\]")
tem_ramp_1_color = '#332288'
tem_ramp_2_color = '#882255'
tem_ramp_3_color = '#332288'
tem_ramp_4_color = '#882255'


def _clean_line(line: str) -> str:
    return line.strip().replace("\ufeff", "")


def _is_unit_line(line: str) -> bool:
    return bool(re.search(r"[A-Za-z]", line)) and not bool(re.search(r"[0-9]e[\+\-]?[0-9]", line))


def _read_block(lines: List[str], start_idx: int):
    idx = start_idx + 1
    step_name = _clean_line(lines[idx])
    idx += 1

    header_line = _clean_line(lines[idx])
    headers = [h.strip() for h in header_line.split("\t") if h.strip()]
    idx += 1

    if idx < len(lines) and _is_unit_line(lines[idx]):
        idx += 1

    buffer: List[str] = []
    while idx < len(lines) and not STEP_HEADER_RE.match(_clean_line(lines[idx])):
        ln = lines[idx].rstrip("\n")
        if ln.strip():
            buffer.append(ln)
        idx += 1

    if buffer:
        df = pd.read_csv(io.StringIO("\n".join(buffer)), sep="\t", names=headers, engine="python")
    else:
        df = pd.DataFrame(columns=headers)

    low = step_name.lower()
    if "temperature ramp" in low:
        step_type = "temperature_ramp"
    elif "flow sweep" in low:
        step_type = "flow_sweep"
    else:
        step_type = step_name

    return {"name": step_name, "type": step_type, "data": df}, idx


def parse_rheology_file(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text(errors="ignore").splitlines()
    steps: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        if STEP_HEADER_RE.match(_clean_line(lines[idx])):
            step, idx = _read_block(lines, idx)
            steps.append(step)
        else:
            idx += 1
    return steps


def plot_temperature_ramps_grouped(steps: List[Dict[str, Any]]):
    """
    For each temperature ramp, plot G', G'', and Tan(delta) (G''/G') on the same axes (log y).
    Plot ramps 1 and 2 together, and ramps 3 and 4 together, in separate figures.
    """
    # Filter only temperature ramp steps
    ramp_steps = [step for step in steps if step["type"] == "temperature_ramp"]
    # Colors for each ramp
    ramp_colors = {
        0: tem_ramp_1_color,  # First ramp
        1: tem_ramp_2_color,  # Second ramp
        2: tem_ramp_3_color,  # Third ramp
        3: tem_ramp_4_color   # Fourth ramp
    }
    # Line styles for different quantities
    styles = {
        'G\'': '-',      # Solid line for G'
        'G\'\'': '--',   # Dashed line for G''
        'Tan(δ)': ':'    # Dotted line for Tan(δ)
    }

    # Group: [0,1], [2,3]
    groups = [ramp_steps[i:i+2] for i in range(0, len(ramp_steps), 2)]
    figs = []
    for group_idx, group in enumerate(groups):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax2 = ax.twinx()
        mod_handles = []
        tan_handles = []
        for i, step in enumerate(group):
            ramp_idx = group_idx * 2 + i
            color = ramp_colors.get(ramp_idx, '#000000')
            alpha = 0.8
            df = step["data"].copy()
            if {"Temperature", "Storage modulus", "Loss modulus"}.issubset(df.columns):
                tan_delta = (df["Loss modulus"] / df["Storage modulus"]).replace([np.inf, -np.inf], np.nan)
                h1, = ax.plot(df["Temperature"], df["Storage modulus"],
                              color=color, ls=styles['G\''], alpha=alpha,
                              label=f"G' ({step['name']})")
                h2, = ax.plot(df["Temperature"], df["Loss modulus"],
                              color=color, ls=styles['G\'\''], alpha=alpha,
                              label=f"G'' ({step['name']})")
                h3, = ax2.plot(df["Temperature"], tan_delta,
                               color=color, ls=styles['Tan(δ)'], alpha=alpha,
                               label=f"Tan(δ) ({step['name']})")
                mod_handles.extend([h1, h2])
                tan_handles.append(h3)
        ax.set_xlabel("Temperature / °C")
        ax.set_ylabel("Storage Modulus G' / MPa     Loss Modulus G'' / MPa")
        ax2.set_ylabel('Tan(δ)')
        ax.set_yscale("log", nonpositive="mask")
        ax2.set_yscale("log", nonpositive="mask")
        handles = mod_handles + tan_handles
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, fontsize="small", ncol=2, loc='lower left')
        figs.append(fig)
    return figs


def plot_flow_sweeps(steps: List[Dict[str, Any]], ax: plt.Axes):
    temps = []
    sweeps = []
    for step in steps:
        if step["type"] != "flow_sweep":
            continue
        df = step["data"]
        if "Temperature" not in df.columns or "Shear rate" not in df.columns or "Viscosity" not in df.columns:
            continue
        temp = float(df["Temperature"].median()) if not df["Temperature"].empty else float("nan")
        temps.append(temp)
        sweeps.append((temp, df))

    if not sweeps:
        return

    norm = mpl.colors.Normalize(vmin=min(temps), vmax=max(temps))
    cmap = mpl.colormaps.get_cmap("coolwarm")

    for temp, df in sweeps:
        ax.loglog(df["Shear rate"], df["Viscosity"], nonpositive="mask", color=cmap(norm(temp)), label=f"{temp:.0f}°C")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature / °C")

    ax.set_xlabel("Shear rate / $s^{-1}$")
    ax.set_ylabel("Viscosity / $Pa·s$")
    # ax.set_title("Flow sweep: Viscosity vs Shear rate")
    # ax.grid(True, which="both", ls=":")


def main():
    path = Path(txt_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    steps = parse_rheology_file(path)

    # Plot temperature ramp groups (G', G'', Tan(delta)), log y
    ramp_figs = plot_temperature_ramps_grouped(steps)
    ramp_outs = []
    for i, fig in enumerate(ramp_figs, 1):
        out = path.parent / (f"{path.stem}_temperature_ramp_group{i}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        ramp_outs.append(out)

    # Flow sweep plot (as before)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_flow_sweeps(steps, ax2)
    fig2.tight_layout()
    out2 = path.parent / (path.stem + "_flow_sweep.png")
    fig2.savefig(out2, dpi=300)

    print("Saved plots:")
    for out in ramp_outs:
        print(f"  {out}")
    print(f"  {out2}")


if __name__ == "__main__":
    main() 