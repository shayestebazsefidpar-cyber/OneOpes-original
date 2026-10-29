"""Charts of a block run, one panel per atom: FP against time, and g(r) per block."""
from pathlib import Path

import altair as alt
import xarray as xr

from oneopes.block_run import convergence_summary

alt.data_transformers.disable_max_rows()

PANEL_SIZE = {"width": 160, "height": 110}
PANELS_PER_ROW = 6
G_MAX_R_NM = 1.0
"""g(r) is drawn up to this distance."""
G_POINTS = 100
"""About how many points each g(r) curve is drawn with; neighbouring bins are averaged."""


def per_atom(chart: alt.Chart | alt.LayerChart, run: xr.Dataset) -> alt.FacetChart:
    """``chart`` repeated in one panel per atom of ``run``, in the run's atom order."""
    atoms = alt.Facet("atom:N", sort=[str(atom) for atom in run.atom.values], title=None)
    return chart.properties(**PANEL_SIZE).facet(atoms, columns=PANELS_PER_ROW)


def fp_chart(run: xr.Dataset) -> alt.FacetChart:
    """FP against the blocks' mid time; a dashed line marks the end of the converged block."""
    data = run.fp.to_dataframe().reset_index()[["atom", "block", "t_mid_ns", "fp"]]
    fp = alt.Chart().mark_line(point=True).encode(
        x=alt.X("t_mid_ns:Q", title="time (ns)"),
        y=alt.Y("fp:Q", title="FP", scale=alt.Scale(zero=False)),
        tooltip=["block:O", "t_mid_ns:Q", "fp:Q"],
    )
    layers = [fp]
    converged_ns = convergence_summary(run).converged_ns
    if converged_ns is not None:
        one_per_panel = alt.Chart().transform_aggregate(groupby=["atom"], points="count()")
        layers.append(one_per_panel.mark_rule(strokeDash=[4, 4], color="grey").encode(x=alt.datum(converged_ns)))
    return per_atom(alt.LayerChart(layer=layers, data=data), run).resolve_scale(y="independent")


def g_chart(run: xr.Dataset) -> alt.FacetChart:
    """g(r) of each block, coloured from the first block to the last."""
    g = run.g.sel(r_nm=slice(None, G_MAX_R_NM))
    g = g.coarsen(r_nm=max(1, g.sizes["r_nm"] // G_POINTS), boundary="trim").mean()
    data = g.to_dataframe().reset_index()[["atom", "block", "r_nm", "g"]]
    chart = alt.Chart(data).mark_line(strokeWidth=1).encode(
        x=alt.X("r_nm:Q", title="r (nm) to water O"),
        y=alt.Y("g:Q", title="g(r)"),
        color=alt.Color("block:O", scale=alt.Scale(scheme="viridis")),
    )
    return per_atom(chart, run)


def write_charts(run: xr.Dataset, outdir: Path) -> None:
    """Write fp_vs_time.png and g_per_block.png."""
    fp_chart(run).save(outdir / "fp_vs_time.png", scale_factor=2)
    g_chart(run).save(outdir / "g_per_block.png", scale_factor=2)
