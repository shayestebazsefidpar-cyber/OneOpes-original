"""
Shared MDAnalysis atom-selection helpers.
"""

import MDAnalysis as mda


def select_heavy_atoms(u: mda.Universe, resname: str, topology: str) -> mda.AtomGroup:
    """Heavy atoms of residue `resname` in Universe `u`."""
    sel = u.select_atoms(f"resname {resname} and not name H*")
    if len(sel) == 0:
        raise SystemExit(
            f"No heavy atoms found for resname '{resname}' in {topology}. "
            "Check the resname against your own topology."
        )
    return sel


def select_water_oxygens(
    u: mda.Universe,
    topology: str,
    water_resname: str | None = None,
    water_atom_name: str | None = None,
) -> mda.AtomGroup:
    """Water oxygen atoms in Universe `u`.

    By default water residues are identified with MDAnalysis's `water`
    selection keyword. Pass `water_resname` and/or `water_atom_name`
    to select explicitly when the topology uses nonstandard naming.
    """
    residue_sel = f"resname {water_resname}" if water_resname else "water"
    name_sel = f"name {water_atom_name}" if water_atom_name else "name O*"
    sel = u.select_atoms(f"{residue_sel} and {name_sel}")
    if len(sel) == 0:
        raise SystemExit(
            f"No water oxygens found in {topology} (selection: '{residue_sel} and {name_sel}'). "
            "If your topology uses nonstandard water naming, pass the water resname / "
            "oxygen atom name explicitly."
        )
    return sel
