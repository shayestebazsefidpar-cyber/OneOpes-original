import marimo

__generated_with = "0.24.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Funnel axis from BioSimSpace

    Rebuilds the funnel CV atoms (`atoms0`, `atoms1`) of one system with BioSimSpace's
    `makeFunnel()`, with no P0/P1 residues taken from an existing `plumed.dat`.

    `whole.pdb` has the positions but no molecule boundaries, which `makeFunnel()` needs to
    find the protein (largest molecule) and the ligand (second largest); `top.top` + `npt.gro`
    have the boundaries. `read_system` reads the topology with the positions of `whole.pdb`.
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
    system_dir = mo.ui.text(
        value="original/HSP90/OneOPES/lig1",
        label="System directory (containing top.top, npt.gro, whole.pdb)",
        full_width=True,
    )
    system_dir
    return (system_dir,)


@app.cell
def _(Path, molecule_sizes, read_system, system_dir):
    directory = Path(system_dir.value)
    system = read_system(directory / "top.top", directory / "npt.gro", structure=directory / "whole.pdb")
    sizes = molecule_sizes(system)
    print(f"{system.nMolecules()} molecules, {system.nAtoms()} atoms")
    print("protein guess (largest):   {} atoms, molecule {}".format(*sizes[0]))
    print("ligand guess (2nd largest): {} atoms, molecule {}".format(*sizes[1]))
    return (system,)


@app.cell
def _(make_funnel, system):
    funnel = make_funnel(system)
    p0p1 = funnel.p0p1
    print("atoms0 (funnel origin):     ", p0p1.atoms0)
    print("atoms1 (funnel inflection): ", p0p1.atoms1)
    print(f"P0 {p0p1.p0} A, P1 {p0p1.p1} A, axis length {p0p1.axis_length:.2f} A")
    funnel.parameters
    return


if __name__ == "__main__":
    app.run()
