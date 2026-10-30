"""The package source names no specific system and no local path."""
import re
from pathlib import Path

import oneopes

SRC = Path(oneopes.__file__).parent
SYSTEM_SPECIFIC = re.compile(r"HSP90|BRD4|\blig\d|/home/|Escritorio|OneOpes-original", re.IGNORECASE)


def test_the_source_names_no_system_or_local_path():
    hits = [f"{path.relative_to(SRC)}:{number}: {line.strip()}" for path in sorted(SRC.rglob("*.py"))
            for number, line in enumerate(path.read_text().splitlines(), 1) if SYSTEM_SPECIFIC.search(line)]
    assert hits == []
