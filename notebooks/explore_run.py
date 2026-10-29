import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explore a block run

    Opens the `blocks.nc` of `oneopes monitor` or `oneopes protein`: the convergence summary,
    each atom's FP over time, the comparison of each block with the one before, and one
    atom's g(r) per block. Run with `marimo edit notebooks/explore_run.py`, or as a script:
    `python notebooks/explore_run.py -- --run outputs/convergence/blocks.nc`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from oneopes.block_run import load_run, run_facts, run_summary
    from oneopes.convergence import TRANSITION_VARIABLES
    return TRANSITION_VARIABLES, load_run, mo, plt, run_facts, run_summary


@app.cell
def _(mo):
    run_path = mo.ui.text(value=mo.cli_args().get("run") or "outputs/convergence/blocks.nc",
                          label="blocks.nc", full_width=True)
    run_path
    return (run_path,)


@app.cell
def _(load_run, mo, run_facts, run_path, run_summary):
    run = load_run(run_path.value)
    summary = run_summary(run)
    mo.md(f"**{summary.n_blocks} blocks of {summary.block_ns} ns**; converged: {summary.converged}"
          f" (block {summary.converged_block}, {summary.converged_ns} ns). Facts: {run_facts(run)}")
    return (run,)


@app.cell
def _(plt, run):
    fig, ax = plt.subplots(figsize=(7, 4))
    run.fp.plot.line(x="t_mid_ns", hue="atom", marker="o", ax=ax, add_legend=run.sizes["atom"] <= 12)
    ax.set(xlabel="simulation time (ns)", ylabel="FP", title="FP per atom")
    fig
    return


@app.cell
def _(TRANSITION_VARIABLES, run):
    run[TRANSITION_VARIABLES].isel(block=slice(1, None)).to_dataframe()
    return


@app.cell
def _(mo, run):
    atom = mo.ui.dropdown(options=[str(name) for name in run.atom.values], value=str(run.atom.values[0]),
                          label="atom")
    atom
    return (atom,)


@app.cell
def _(atom, plt, run):
    fig_g, ax_g = plt.subplots(figsize=(7, 4))
    run.g.sel(atom=atom.value).plot.line(x="r_nm", hue="block", alpha=0.6, ax=ax_g)
    ax_g.set(xlabel="r (nm) to water O", ylabel="g(r)", xlim=(0, 1.0), title=f"g(r) per block - {atom.value}")
    fig_g
    return


if __name__ == "__main__":
    app.run()
