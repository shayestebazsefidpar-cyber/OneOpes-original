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

--- Atom serial convention (code-review Rule 3) ---
Every `serial` below is 1-BASED, matching the official vendored algorithm's
own printed output (`run_official_selection.py`'s `select_next_atom()`/
`select_bulk_atom()` print `int(atom.index + 1)`) and the ranking CSV's
`atom` column it reads back in (see
`official_selection/prepare_ranking_csv.py` and that subpackage's README).
This is DISTINCT from MDAnalysis's own `Atom.index`/`AtomGroup.indices`,
which are 0-based - `serial == mdanalysis_index + 1` for the specific
selection expression used throughout this pipeline
(`resname <ligand> and not name H*`), but the two are not
interchangeable in general and nothing in this module converts between
them - see AtomRef below.

--- Structured types (code-review Rule 3) ---
`AtomRef` and `SelectionResult` (frozen dataclasses) are used internally by
`write_g1_g2_yaml()` for a validated in-process representation of a G1/G2
result. `parse_selection_lines()` still returns the original
`{"G1": [...], "G2": [...]}` dict shape unchanged - changing that would
have no benefit and would break its existing callers/tests for no reason.
`write_g1_g2_yaml()` accepts either that dict OR a `SelectionResult`
(auto-detected), and always WRITES the same YAML shape as before -
`SelectionResult.to_yaml_dict()` is the one place that mapping happens, so
the on-disk schema `ligand_waterfp.ligand_cv.build_ligand_cv` reads is
never affected by this refactor.
"""
import os
import re
from dataclasses import dataclass
import yaml

LINE_RE = re.compile(r"(anti-bulk|bulk) fp selection:\s*(\w+)\s*\((\d+)\),\s*(\w+)\s*\((\d+)\)")


@dataclass(frozen=True)
class AtomRef:
    """One atom's (name, serial) pair as used throughout the G1/G2 result.

    serial is 1-BASED - see this module's docstring for exactly what that
    means and how it relates to (and differs from) MDAnalysis's own
    0-based Atom.index. Not changed by this refactor - AtomRef only gives
    the existing (name, serial) pair a real type instead of a 2-key dict.
    """
    name: str
    serial: int

    @classmethod
    def from_dict(cls, d):
        return cls(name=d["name"], serial=int(d["serial"]))

    def to_dict(self):
        return {"name": self.name, "serial": self.serial}


@dataclass(frozen=True)
class SelectionResult:
    """A validated, in-process G1/G2 result: exactly 2 AtomRefs per group,
    same shape `parse_selection_lines()` has always produced, plus an
    optional system_id. Serializes to (and parses from) the existing
    dict/YAML shape via to_yaml_dict()/from_dict() - see this module's
    docstring for why the YAML schema itself never changes.
    """
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
        """The exact dict shape write_g1_g2_yaml() has always written:
        {"G1": [{"name":.., "serial":..}, ...], "G2": [...]} plus
        "system_id" only if set - never dataclasses.asdict() directly,
        since that would still be correct here (AtomRef's own field names
        already match the YAML schema) but this makes the mapping explicit
        rather than incidental, and mirrors StabilityResult's convention
        in monitor_convergence.py.
        """
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
    """Write a G1/G2 result to a YAML file, creating the parent directory
    if needed. The exact same YAML shape as always:
    {"G1": [{"name":.., "serial":..}, ...], "G2": [...] [, "system_id":..]}.

    `result` may be either the original {'G1': [...], 'G2': [...]} dict
    (existing callers/tests keep working unchanged) or a `SelectionResult`
    - auto-detected below. Assumes a dict `result` already has both keys -
    callers should validate that (see
    official_selection.run_official_selection.main) before calling this,
    since the appropriate error message depends on what was actually
    parsed.
    """
    if isinstance(result, SelectionResult):
        payload = result.to_yaml_dict()
        if system_id:
            payload["system_id"] = system_id
    else:
        payload = {"G1": result["G1"], "G2": result["G2"]}
        if system_id:
            payload["system_id"] = system_id

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
