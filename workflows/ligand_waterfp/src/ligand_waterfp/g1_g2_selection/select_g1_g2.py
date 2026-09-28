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

--- Atom serial convention ---
`serial` is 1-based, matching the official algorithm's own printed output
and the ranking CSV's `atom` column - NOT MDAnalysis's 0-based
`Atom.index` (serial == mdanalysis_index + 1 for the selection used here).

--- AtomRef / SelectionResult ---
Frozen dataclasses giving this module's {"G1": [...], "G2": [...]} shape
a validated in-process type. `parse_selection_lines()` still returns the
original dict; `write_g1_g2_yaml()` accepts either and always writes the
same YAML.
"""
import re
from dataclasses import dataclass
from pathlib import Path
import yaml

LINE_RE = re.compile(r"(anti-bulk|bulk) fp selection:\s*(\w+)\s*\((\d+)\),\s*(\w+)\s*\((\d+)\)")


@dataclass(frozen=True)
class AtomRef:
    """One atom's (name, serial) pair. serial is 1-based - see module
    docstring; do not confuse with MDAnalysis's 0-based Atom.index."""
    name: str
    serial: int

    @classmethod
    def from_dict(cls, d):
        return cls(name=d["name"], serial=int(d["serial"]))

    def to_dict(self):
        return {"name": self.name, "serial": self.serial}


@dataclass(frozen=True)
class SelectionResult:
    """Structured G1/G2 selection result: 2 AtomRefs per group plus an
    optional system_id. to_yaml_dict()/from_dict() convert to/from the
    existing YAML dict shape (see module docstring)."""
    G1: tuple[AtomRef, AtomRef]
    G2: tuple[AtomRef, AtomRef]
    system_id: str | None = None

    @classmethod
    def from_dict(cls, d):
        return cls(
            G1=tuple(AtomRef.from_dict(a) for a in d["G1"]),
            G2=tuple(AtomRef.from_dict(a) for a in d["G2"]),
            system_id=d.get("system_id"),
        )

    def to_yaml_dict(self):
        """Serialize to the existing YAML dict shape (adds "system_id"
        only if set)."""
        payload = {
            "G1": [a.to_dict() for a in self.G1],
            "G2": [a.to_dict() for a in self.G2],
        }
        if self.system_id:
            payload["system_id"] = self.system_id
        return payload


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
    """Write a G1/G2 result to YAML, creating the parent directory if
    needed. Accepts either the original {'G1': [...], 'G2': [...]} dict
    or a SelectionResult - both produce the same YAML shape. `out_path`
    may be a str or a Path. A dict `result` is assumed to already have
    both keys; callers should validate that first (see
    official_selection.run_official_selection.main).
    """
    if isinstance(result, SelectionResult):
        payload = result.to_yaml_dict()
        if system_id:
            payload["system_id"] = system_id
    else:
        payload = {"G1": result["G1"], "G2": result["G2"]}
        if system_id:
            payload["system_id"] = system_id

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
