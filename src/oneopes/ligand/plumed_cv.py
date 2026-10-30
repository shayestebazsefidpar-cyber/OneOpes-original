"""Starting-point PLUMED fragment for the G1/G2 CVs (``oneopes build-cv``)."""
import logging
from pathlib import Path

from cyclopts.types import ExistingFile

from oneopes.ligand.g1_g2 import read_g1_g2_yaml

logger = logging.getLogger(__name__)

FRAGMENT_TEMPLATE = """\
# Ligand-side WaterFP CVs, written by `oneopes build-cv` from g1_g2.yaml.
# G1: anti-bulk atom pair (most perturbed water), G2: bulk atom pair (most bulk-like water).
# Starting point only: review R_0, NN, MM and the WATER_O group before production.

G1: GROUP ATOMS={g1_serials}
G2: GROUP ATOMS={g2_serials}

{water_group_line}

g1_coord: COORDINATION GROUPA=G1 GROUPB=WATER_O R_0=0.32 NN=6 MM=12
g2_coord: COORDINATION GROUPA=G2 GROUPB=WATER_O R_0=0.32 NN=6 MM=12

PRINT ARG=g1_coord,g2_coord STRIDE=500 FILE=LIGAND_WATERFP_CV.dat
"""


def build_cv(*, g1_g2: ExistingFile, out: Path, water_group_def: str | None = None) -> None:
    """Write a starting-point PLUMED fragment with GROUP and COORDINATION CVs for G1 and G2.

    Not a ready-to-run plumed.dat: the switching-function parameters (R_0, NN, MM) are placeholders
    to review before production.

    Parameters
    ----------
    g1_g2
        g1_g2.yaml written by `oneopes select` or `oneopes monitor`.
    out
        The PLUMED fragment to write.
    water_group_def
        PLUMED GROUP body of the water-oxygen reference group, e.g. 'ATOMS=1305-14858:3' (every
        third atom: the oxygens of a 3-site water); without it, the fragment refers to a WATER_O
        group you define yourself.
    """
    selection = read_g1_g2_yaml(g1_g2)

    g1_serials = ",".join(str(a.serial) for a in selection.G1)
    g2_serials = ",".join(str(a.serial) for a in selection.G2)

    if water_group_def:
        water_group_line = f"WATER_O: GROUP {water_group_def}"
    else:
        water_group_line = ("# TODO: define your own water-oxygen reference group, e.g.:\n"
                            "# WATER_O: GROUP ATOMS=<water oxygen atom indices>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FRAGMENT_TEMPLATE.format(g1_serials=g1_serials, g2_serials=g2_serials,
                                            water_group_line=water_group_line))

    logger.info("G1 atoms (serials): %s; G2 atoms (serials): %s; wrote %s", g1_serials, g2_serials, out.resolve())
