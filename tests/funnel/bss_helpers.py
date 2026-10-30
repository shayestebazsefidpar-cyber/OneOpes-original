"""Helpers for reading BioSimSpace systems in the funnel tests."""


def molecule_layout(system):
    """(protein_index, ligand_index, atom offsets, atom counts) by BSS's largest/second-largest rule."""
    counts = [m.nAtoms() for m in system]
    ranked = sorted((n, i) for i, n in enumerate(counts))
    offsets = [sum(counts[:i]) for i in range(len(counts))]
    return ranked[-1][1], ranked[-2][1], offsets, counts


def bss_xyz(atom):
    """BSS atom coordinates in Angstrom."""
    c = atom.coordinates()
    return [c.x().angstroms().value(), c.y().angstroms().value(), c.z().angstroms().value()]
