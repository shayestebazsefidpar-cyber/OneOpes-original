import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explore a block run

    Opens the `blocks.nc` of `oneopes monitor` or `oneopes protein`: the convergence summary,
    the comparison of each block with the one before, and the chosen atoms' FP over time and
    g(r) per block. Run with `marimo edit notebooks/explore_run.py`, or as a script:
    `python notebooks/explore_run.py -- --run outputs/convergence/blocks.nc`.
    """)
    return


@app.cell
def _():
    import marimo as mo

    from oneopes.block_run import convergence_summary, facts_of, load_run
    from oneopes.convergence import transitions
    from oneopes.plots import fp_chart, g_chart
    return convergence_summary, facts_of, fp_chart, g_chart, load_run, mo, transitions


@app.cell
def _(mo):
    run_path = mo.ui.text(value=mo.cli_args().get("run") or "outputs/convergence/blocks.nc",
                          label="blocks.nc", full_width=True)
    run_path
    return (run_path,)


@app.cell
def _(convergence_summary, facts_of, load_run, mo, run_path):
    run = load_run(run_path.value)
    summary = convergence_summary(run)
    mo.md(f"**{summary.n_blocks} blocks of {summary.block_ns} ns**; converged: {summary.converged}"
          f" (block {summary.converged_block}, {summary.converged_ns} ns). Facts: {facts_of(run)}")
    return (run,)


@app.cell
def _(run, transitions):
    transitions(run).to_dataframe()
    return


@app.cell
def _(mo, run):
    atom_names = [str(name) for name in run.atom.values]
    atoms = mo.ui.multiselect(options=atom_names, value=atom_names[:6], label="atoms")
    atoms
    return (atoms,)


@app.cell
def _(atoms, fp_chart, g_chart, mo, run):
    chosen = run.sel(atom=atoms.value)
    mo.vstack([fp_chart(chosen), g_chart(chosen)])
    return


if __name__ == "__main__":
    app.run()
