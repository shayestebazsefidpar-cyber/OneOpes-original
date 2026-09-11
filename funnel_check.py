import marimo

__generated_with = "0.24.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Independent funnel-axis check with BioSimSpace (HSP90 lig1)

    Reconstructs the funnel collective-variable definition (`atoms0`, `atoms1`)
    directly from the HSP90 `lig1` `whole.pdb` structure using
    `BioSimSpace.Metadynamics.CollectiveVariable.makeFunnel()`, with **no**
    hard-coded p0/p1 residues from `plumed.dat` or `select_hydration_cvs.py`.

    `makeFunnel()` auto-detects protein (largest molecule) and ligand (second
    largest molecule) when `protein=`/`ligand=` are not given, so the only job
    here is to hand it a `BioSimSpace.System` with correct molecule boundaries.

    Caveat: `whole.pdb` alone (as dumped by `gmx trjconv`) has no `TER` records,
    so BioSimSpace/Sire parses it as a single-molecule blob and protein/ligand
    auto-detection can't work. The topology (`top.top` + `npt.gro`) gives
    correct molecule identity, but its water/ion atom names don't match
    `whole.pdb`'s, so the two can't be loaded together directly. The fix:
    build the system from `top.top` + `npt.gro` (correct topology), then
    transplant `whole.pdb`'s actual coordinates onto it by atom index (both
    files share the same atom count and order).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import sire as sr
    import BioSimSpace as BSS
    import MDAnalysis as mda
    from BioSimSpace.Metadynamics.CollectiveVariable import makeFunnel

    return BSS, makeFunnel, mda, mo, sr


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
def _(BSS, system_dir):
    top_file = f"{system_dir.value}/top.top"
    gro_file = f"{system_dir.value}/npt.gro"
    pdb_file = f"{system_dir.value}/whole.pdb"

    # Topology + starting coordinates: gives correct per-molecule identity
    # (protein, ligand "MOL", ions, waters), unlike whole.pdb on its own.
    topo_system = BSS.IO.readMolecules([top_file, gro_file])
    print(f"Molecules: {topo_system.nMolecules()}, atoms: {topo_system.nAtoms()}")
    return pdb_file, topo_system


@app.cell
def _(BSS, mda, pdb_file, sr, topo_system):
    # Overwrite coordinates with the actual whole.pdb frame, atom-index matched
    # (same count/order as npt.gro since both come from the same GROMACS run).
    u = mda.Universe(pdb_file)
    coords = u.atoms.positions.astype(float)
    assert len(coords) == topo_system.nAtoms(), "atom count mismatch between whole.pdb and topology"

    modern_sys = sr.system.System(topo_system._sire_object)
    cursor = modern_sys.cursor()
    for atom_cursor, xyz in zip(cursor.atoms(), coords):
        atom_cursor["coordinates"] = sr.legacy.Maths.Vector(*xyz.tolist())
    committed = cursor.commit()

    legacy_system = committed._system if hasattr(committed, "_system") else committed
    system = BSS._SireWrappers.System(legacy_system)
    return (system,)


@app.cell
def _(system):
    # Sanity check: confirm the auto-detect heuristic (largest = protein,
    # second largest = ligand) picks sensible molecules for this system.
    sizes = sorted(
        ((m.nAtoms(), i) for i, m in enumerate(system.getMolecules())), reverse=True
    )
    print("Largest molecule (protein guess):  natoms={}, idx={}".format(*sizes[0]))
    print("2nd largest molecule (ligand guess): natoms={}, idx={}".format(*sizes[1]))
    return


@app.cell
def _(makeFunnel, system):
    # No protein=/ligand= passed: auto-detected from molecule size, and no
    # p0/p1 residue lists reused from plumed.dat or select_hydration_cvs.py.
    atoms0, atoms1 = makeFunnel(system)
    print("atoms0 (funnel origin):     ", atoms0)
    print("atoms1 (funnel inflection): ", atoms1)
    return


if __name__ == "__main__":
    app.run()
