"""Figures of a block run: FP against time, and g(r) per block."""
import xarray as xr
from matplotlib.figure import Figure


def fp_figure(run: xr.Dataset, atom_name: str | None = None) -> Figure:
    """FP against the blocks' mid time, of one atom or, without ``atom_name``, of every atom."""
    fig = Figure(figsize=(5, 3.5), layout="tight")
    ax = fig.subplots()
    if atom_name is None:
        run.fp.plot.line(x="t_mid_ns", hue="atom", marker="o", ax=ax, add_legend=run.sizes["atom"] <= 12)
    else:
        run.fp.sel(atom=atom_name).plot.line(x="t_mid_ns", marker="o", ax=ax)
    ax.set(xlabel="simulation time (ns)", ylabel="FP (WaterFP excess-entropy integral)",
           title=f"Hydration FP vs time - {atom_name or 'every atom'}")
    return fig


def g_figure(run: xr.Dataset, atom_name: str) -> Figure:
    """One atom's g(r) in each block, with a legend up to eight blocks."""
    fig = Figure(figsize=(5, 3.5), layout="tight")
    ax = fig.subplots()
    run.g.sel(atom=atom_name).plot.line(x="r_nm", hue="block", alpha=0.6, ax=ax, add_legend=run.sizes["block"] <= 8)
    ax.set(xlabel="r (nm) to water O", ylabel="g(r)", xlim=(0, 1.0), title=f"RDF per block - {atom_name}")
    return fig
