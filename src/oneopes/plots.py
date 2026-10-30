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
    """``chart`` repeated in one panel per atom of ``run``, in the run's atom order, each with its own x axis."""
    atoms = alt.Facet("atom:N", sort=[str(atom) for atom in run.atom.values], title=None)
    return chart.properties(**PANEL_SIZE).facet(atoms, columns=PANELS_PER_ROW).resolve_axis(x="independent")


def time_rule(ns: float, stroke_dash: list[int]) -> alt.Chart:
    """A grey vertical line at ``ns`` in each panel."""
    one_per_panel = alt.Chart().transform_aggregate(groupby=["atom"], points="count()")
    return one_per_panel.mark_rule(strokeDash=stroke_dash, color="grey").encode(x=alt.datum(ns))


def fp_chart(run: xr.Dataset) -> alt.FacetChart:
    """Each atom's FP against the blocks' mid time, as a change from its mean, on one shared scale.

    A relative change keeps atoms with very different FPs comparable. A solid line marks where
    production starts, as the last check found it, and a dashed line the end of the converged block.
    """
    data = run.fp.to_dataframe().reset_index()[["atom", "block", "t_mid_ns", "fp"]]
    mean_fp = data.groupby("atom")["fp"].transform("mean")
    data["fp_change_pct"] = 100 * (data["fp"] - mean_fp) / mean_fp.abs()
    fp = alt.Chart().mark_line(point=True).encode(
        x=alt.X("t_mid_ns:Q", title="time (ns)"),
        y=alt.Y("fp_change_pct:Q", title="FP vs mean (%)"),
        tooltip=["block:O", "t_mid_ns:Q", "fp:Q", alt.Tooltip("fp_change_pct:Q", format=".1f")],
    )
    layers = [fp]
    summary = convergence_summary(run)
    if summary.production_start_ns is not None:
        layers.append(time_rule(summary.production_start_ns, []))
    if summary.converged_ns is not None:
        layers.append(time_rule(summary.converged_ns, [4, 4]))
    return per_atom(alt.LayerChart(layer=layers, data=data), run)


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
