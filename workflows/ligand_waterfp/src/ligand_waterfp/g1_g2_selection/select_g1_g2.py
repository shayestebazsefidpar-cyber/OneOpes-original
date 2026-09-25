"""
Turns the official WaterFP selection algorithm's raw text output into a
small, structured G1/G2 result - the format
`ligand_waterfp.ligand_cv.build_ligand_cv` consumes to construct the
ligand-side PLUMED CV.

This module has no CLI of its own. `parse_selection_lines()` and
`write_g1_g2_yaml()` are called directly, in-process, by
`ligand_waterfp.official_selection.run_official_selection`'s `main()`
immediately after the vendored `select_next_atom()`/`select_bulk_atom()`
return their result sentences - there is no intermediate text file and no
separate parsing stage anymore. Kept as pure functions here (rather than
inlined into official_selection) so they stay independently testable with
synthetic text, with no dependency on a real .tpr/ranking CSV.

    "anti-bulk fp selection: ATOM1 (5), ATOM2 (12)"  -> G1 = [(ATOM1, 5), (ATOM2, 12)]
    "bulk fp selection: ATOM3 (18), ATOM4 (19)"      -> G2 = [(ATOM3, 18), (ATOM4, 19)]
"""
import os
import re
import yaml

LINE_RE = re.compile(r"(anti-bulk|bulk) fp selection:\s*(\w+)\s*\((\d+)\),\s*(\w+)\s*\((\d+)\)")


def parse_selection_lines(text):
    """Returns {'G1': [{'name':..,'serial':..}, ...], 'G2': [...]}."""
    result = {}
    for m in LINE_RE.finditer(text):
        kind, name1, serial1, name2, serial2 = m.groups()
        key = "G1" if kind == "anti-bulk" else "G2"
        result[key] = [
            {"name": name1, "serial": int(serial1)},
            {"name": name2, "serial": int(serial2)},
        ]
    return result


def write_g1_g2_yaml(result, out_path, system_id=None):
    """Write a parsed {'G1': [...], 'G2': [...]} result to a YAML file,
    creating the parent directory if needed. Assumes result already has
    both keys - callers should validate that (see
    official_selection.run_official_selection.main) before calling this,
    since the appropriate error message depends on what was actually
    parsed."""
    payload = {"G1": result["G1"], "G2": result["G2"]}
    if system_id:
        payload["system_id"] = system_id

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
