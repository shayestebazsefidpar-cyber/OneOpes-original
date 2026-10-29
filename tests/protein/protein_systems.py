"""Synthetic test systems (made-up names, coordinates and sizes; unrelated to any real system)."""

import numpy as np

MASSES = {"N": 14.007, "C": 12.011, "O": 15.999, "S": 32.06, "H": 1.008, "Na": 22.99}


def build_universe(protein_atoms, ligand_atoms, n_waters=0, box=60.0, rng_seed=7, extra_residues=()):
    """Build an MDAnalysis Universe.

    protein_atoms: list of (resname, resid, atom_name, element, xyz)
    ligand_atoms:  list of (atom_name, element, xyz) in one residue named 'QZX'
    extra_residues: list of (resname, [(atom_name, element, xyz), ...]) (e.g. ions)
    Waters 'SOL' (OW/HW1/HW2) are placed at random, at least 4 A from protein/ligand.
    """
    import MDAnalysis as mda

    rng = np.random.default_rng(rng_seed)
    atoms = []  # (resname, resid, name, element, xyz, res_key)
    for resname, resid, name, element, xyz in protein_atoms:
        atoms.append((resname, resid, name, element, xyz, ("P", resid)))
    lig_resid = 900
    for name, element, xyz in ligand_atoms:
        atoms.append(("QZX", lig_resid, name, element, xyz, ("L", lig_resid)))
    for k, (resname, ratoms) in enumerate(extra_residues):
        for name, element, xyz in ratoms:
            atoms.append((resname, 950 + k, name, element, xyz, ("X", k)))
    solute = np.array([a[4] for a in atoms], dtype=float)
    placed = 0
    while placed < n_waters:
        o = rng.uniform(0, box, 3)
        if len(solute) and np.min(np.linalg.norm(solute - o, axis=1)) < 4.0:
            continue
        rid = 1000 + placed
        atoms.append(("SOL", rid, "OW", "O", o, ("W", rid)))
        atoms.append(("SOL", rid, "HW1", "H", o + [0.96, 0, 0], ("W", rid)))
        atoms.append(("SOL", rid, "HW2", "H", o + [-0.24, 0.93, 0], ("W", rid)))
        placed += 1

    keys = list(dict.fromkeys(a[5] for a in atoms))
    resindex = [keys.index(a[5]) for a in atoms]
    u = mda.Universe.empty(len(atoms), n_residues=len(keys), atom_resindex=resindex, trajectory=True)
    first = {}
    for a in atoms:
        first.setdefault(a[5], a)
    u.add_TopologyAttr("names", [a[2] for a in atoms])
    u.add_TopologyAttr("elements", [a[3] for a in atoms])
    u.add_TopologyAttr("types", [a[3] for a in atoms])
    u.add_TopologyAttr("masses", [MASSES[a[3]] for a in atoms])
    u.add_TopologyAttr("resnames", [first[k][0] for k in keys])
    u.add_TopologyAttr("resids", [first[k][1] for k in keys])
    u.add_TopologyAttr("chainIDs", ["B"] * len(atoms))
    u.atoms.positions = np.array([a[4] for a in atoms], dtype=float)
    u.dimensions = [box, box, box, 90, 90, 90]
    return u


def simple_protein(n_res=12, origin=(10.0, 10.0, 10.0), spacing=3.5, direction=(1.0, 0.0, 0.0)):
    """A straight chain of ALA residues (N, CA, C, O, CB, H) along ``direction``."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    perp = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    out = []
    for i in range(n_res):
        base = np.asarray(origin, float) + i * spacing * d
        out += [
            ("ALA", 20 + i, "N", "N", base),
            ("ALA", 20 + i, "CA", "C", base + 1.2 * perp),
            ("ALA", 20 + i, "C", "C", base + 1.2 * perp + 1.0 * d),
            ("ALA", 20 + i, "O", "O", base + 2.3 * perp + 1.0 * d),
            ("ALA", 20 + i, "CB", "C", base + 1.2 * perp + 1.4 * np.cross(d, perp)),
            ("ALA", 20 + i, "H", "H", base - 1.0 * perp),
        ]
    return out


def protein_args(directory) -> list[str]:
    """`oneopes protein` with its required inputs as (empty) files in ``directory``."""
    for name in ("c.pdb", "t.tpr", "x.xtc"):
        (directory / name).touch()
    return ["protein", "--complex", str(directory / "c.pdb"), "--topology", str(directory / "t.tpr"),
            "--trajectory", str(directory / "x.xtc"), "--output-dir", str(directory / "o")]

