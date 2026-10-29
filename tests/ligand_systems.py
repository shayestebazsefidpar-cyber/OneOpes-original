"""Synthetic ligand-in-water systems written to files, for end-to-end tests of the ligand commands."""
import MDAnalysis as mda
import numpy as np


def ligand_in_water(directory, n_frames: int = 3) -> tuple[str, str]:
    """A 6-atom chain ligand (resname MOL, C0-C5) in a 5 nm box of randomly placed water oxygens.

    Frames are 1 ps apart. Returns the paths of the topology (PDB with bonds) and the trajectory (XTC).
    """
    rng = np.random.default_rng(1)
    box, n_lig, n_water = 50.0, 6, 4000
    chain = np.array([[25.0 + 1.5 * i, 25.0, 25.0] for i in range(n_lig)])
    frames = [np.vstack([chain, rng.uniform(0.0, box, (n_water, 3))]) for _ in range(n_frames)]

    u = mda.Universe.empty(n_lig + n_water, n_residues=1 + n_water,
                           atom_resindex=[0] * n_lig + list(range(1, n_water + 1)), trajectory=True)
    u.add_TopologyAttr("names", [f"C{i}" for i in range(n_lig)] + ["O"] * n_water)
    u.add_TopologyAttr("resnames", ["MOL"] + ["SOL"] * n_water)
    u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(n_lig - 1)])
    u.load_new(np.array(frames), order="fac", dimensions=[box, box, box, 90, 90, 90])

    top, traj = str(directory / "top.pdb"), str(directory / "traj.xtc")
    u.atoms.write(top)
    with mda.Writer(traj, u.atoms.n_atoms) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return top, traj
