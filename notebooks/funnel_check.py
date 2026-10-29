import marimo

__generated_with = "0.24.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Funnel axis from BioSimSpace

    The funnel CV atoms (`atoms0`, `atoms1`) and axis of one system, from BioSimSpace's `makeFunnel()`.

    `whole.pdb` has the positions but no molecule boundaries, which `makeFunnel()` needs to
    find the protein (largest molecule) and the ligand (second largest); `top.top` + `npt.gro`
    have the boundaries. `read_system` reads the topology with the positions of `whole.pdb`.
    Run with `marimo edit notebooks/funnel_check.py`, or as a script:
    `python notebooks/funnel_check.py -- --dir path/to/system`.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from oneopes.funnel.bss import molecule_sizes, read_system
    from oneopes.funnel.make_funnel import make_funnel
    return Path, make_funnel, mo, molecule_sizes, read_system


@app.cell
def _(mo):
    system_dir = mo.ui.text(value=mo.cli_args().get("dir") or "",
                            label="System directory (containing top.top, npt.gro, whole.pdb)", full_width=True)
    system_dir
    return (system_dir,)


@app.cell
def _(Path, mo, molecule_sizes, read_system, system_dir):
    mo.stop(not system_dir.value, mo.md("Enter a system directory."))
    directory = Path(system_dir.value)
    system = read_system(directory / "top.top", directory / "npt.gro", structure=directory / "whole.pdb")
    (protein_atoms, protein), (ligand_atoms, ligand) = molecule_sizes(system)[:2]
    mo.md(f"{system.nMolecules()} molecules, {system.nAtoms()} atoms. Protein (largest): molecule {protein}, "
          f"{protein_atoms} atoms; ligand (second largest): molecule {ligand}, {ligand_atoms} atoms.")
    return (system,)


@app.cell
def _(make_funnel, mo, system):
    funnel = make_funnel(system)
    axis = funnel.axis
    mo.vstack([
        mo.md(f"atoms0 (funnel origin): {list(axis.atoms0)}  \natoms1 (funnel inflection): {list(axis.atoms1)}  \n"
              f"P0 {axis.p0} A, P1 {axis.p1} A, axis length {axis.length:.2f} A"),
        funnel.parameters,
    ])
    return


if __name__ == "__main__":
    app.run()
