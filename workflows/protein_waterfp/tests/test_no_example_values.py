"""The package source holds no PLUMED / funnel / Zone code and no example-specific systems, paths or values."""

import re
from pathlib import Path

import protein_waterfp

SRC = Path(protein_waterfp.__file__).parent

FORBIDDEN = [
    r"(?i)plumed", r"(?i)\bzone", r"(?i)funnel", r"(?i)deep", r"(?i)entrance", r"oneopes_funnel", r"BRD4", r"HSP90", r"\blig\d", r"\bLIG\b", r"\bMOL\b", r"run0", r"rep_\d", r"WaterFP_protein_Ligand",
    r"/home/", r"Escritorio", r"parents\[", r"\bglob\(", r"\.\./",
    r"\b100(\.0)?\b", r"\b200(\.0)?\b", r"(?i)\d+\s*ns\b", r"(?i)n_blocks\s*=\s*\d", r"(?i)early.?stop",
    r"\b1\.028\b", r"\b1\.368\b", r"\b0\.02\b", r"\b115\b", r"\b1586\b",
]


def test_no_example_specific_values_or_path_discovery_in_source():
    hits = []
    for path in sorted(SRC.glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for pat in FORBIDDEN:
                if re.search(pat, line):
                    hits.append(f"{path.name}:{n}: {pat}: {line.strip()}")
    assert hits == []
