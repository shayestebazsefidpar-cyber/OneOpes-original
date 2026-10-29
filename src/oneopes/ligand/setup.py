"""Build and equilibrate a ligand-in-water GROMACS system (``oneopes prepare-system``)."""
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Literal, get_args

import numpy as np
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number
from scipy.spatial.distance import pdist

from oneopes.errors import InputError
from oneopes.system import load_universe
from oneopes.waterfp import RadialGrid

logger = logging.getLogger(__name__)

Step = Literal["box", "solvate", "ions_tpr", "genion", "em_tpr", "em_run", "nvt_tpr", "nvt_run", "npt_tpr", "npt_run"]
STEPS = get_args(Step)
MDP_DIR = Path(__file__).parent / "mdp"
BoxType = Literal["triclinic", "cubic", "dodecahedron", "octahedron"]
MIN_BOX_NM = round(2 * RadialGrid().rmax_nm + 0.2, 1)
"""Twice WaterFP's default rmax, so no water is counted twice through the periodic box, plus room for NPT."""


def box_edges_nm(ligand_gro: Path, box_type: BoxType, distance_nm: float, min_edge_nm: float) -> list[float]:
    """Box size for `gmx editconf -box`: the ligand with ``distance_nm`` on each side, and at least ``min_edge_nm``.

    The ligand's size is what editconf -d would use: its extent along each axis for a triclinic box,
    its diameter (largest atom-atom distance) otherwise.
    """
    positions_nm = load_universe(ligand_gro).atoms.positions / 10
    sizes = np.ptp(positions_nm, axis=0) if box_type == "triclinic" else [np.max(pdist(positions_nm), initial=0.0)]
    return [round(max(float(size) + 2 * distance_nm, min_edge_nm), 3) for size in sizes]


def run_step(step: Step, cmd: list[str], cwd: Path, stdin: str | None = None) -> None:
    """Run the gmx command of ``step`` in ``cwd``, logging it first; ``stdin`` answers its prompt.

    Raises InputError, naming the step to resume from, when gmx is missing or the command fails.
    """
    shown = shlex.join(cmd) if stdin is None else f"echo {stdin} | {shlex.join(cmd)}"
    logger.info("$ %s", shown)
    try:
        subprocess.run(cmd, cwd=cwd, check=True, input=None if stdin is None else f"{stdin}\n", text=True)
    except FileNotFoundError:
        raise InputError(f"{cmd[0]!r} not found; pass the gmx executable with --gmx.") from None
    except subprocess.CalledProcessError as failure:
        raise InputError(f"{shlex.join(cmd[:2])} failed at step {step} (exit {failure.returncode}; see its output "
                         f"above). Fix it and rerun with --start-from {step}.") from None


def prepare_system(*, ligand_gro: ExistingFile, ligand_top: ExistingFile, outdir: Path, mdp_dir: Path = MDP_DIR,
                   box_distance_nm: Annotated[float, Parameter(validator=Number(gt=0))] = 1.2,
                   min_box_nm: Annotated[float, Parameter(validator=Number(gt=0))] = MIN_BOX_NM,
                   box_type: BoxType = "cubic",
                   water_model: str = "spc216",
                   ion_conc: Annotated[float, Parameter(validator=Number(gte=0))] = 0.0,
                   pname: str = "NA", nname: str = "CL", gmx: str = "gmx",
                   maxwarn: Annotated[int, Parameter(validator=Number(gte=0))] = 0,
                   start_from: Step = "box") -> None:
    """Build and equilibrate a ligand-in-water GROMACS system for the unbiased run the monitor follows.

    Box definition -> solvation -> ionization -> EM -> NVT -> NPT, with no protein present
    (see docs/ligand_setup.md). Production MD is left to you, so that `oneopes monitor` can
    follow the running mdrun's PID; its settings, prod.mdp, are copied to --outdir.

    Parameters
    ----------
    ligand_gro
        Prepared ligand structure (no water or protein).
    ligand_top
        Matching GROMACS topology.
    outdir
        Working directory for all intermediate and output files.
    mdp_dir
        Directory with ions.mdp, em.mdp, nvt.mdp, npt.mdp and prod.mdp.
    box_distance_nm
        Smallest distance (nm) between the ligand and the box edge.
    min_box_nm
        Smallest box edge (nm): twice the WaterFP rmax the monitor will use, plus room for NPT.
    box_type
        Box shape (`gmx editconf -bt`).
    water_model
        Coordinate file for `gmx solvate -cs` (spc216 suits TIP3P-class water models).
    ion_conc
        Ion concentration (mol/L) for genion, on top of neutralization.
    pname
        Name of the positive ion (`gmx genion -pname`).
    nname
        Name of the negative ion (`gmx genion -nname`).
    gmx
        gmx executable.
    maxwarn
        GROMACS warnings grompp may ignore for EM, NVT and NPT. ions.tpr always allows the one
        warning of a not yet neutral system.
    start_from
        Resume from this step instead of the beginning.
    """
    mdp_dir = mdp_dir.resolve()
    for name in ("ions.mdp", "em.mdp", "nvt.mdp", "npt.mdp", "prod.mdp"):
        if not (mdp_dir / name).is_file():
            raise InputError(f"Required mdp file not found: {mdp_dir / name}")
    outdir.mkdir(parents=True, exist_ok=True)
    if not (outdir / "system.top").exists():
        shutil.copy(ligand_top, outdir / "system.top")
    if not (outdir / "prod.mdp").exists():
        shutil.copy(mdp_dir / "prod.mdp", outdir / "prod.mdp")

    def grompp(mdp: str, structure: list[str], tpr: str, allowed_warnings: int = maxwarn) -> list[str]:
        return [gmx, "grompp", "-f", str(mdp_dir / mdp), "-c", *structure, "-p", "system.top", "-o", tpr,
                "-maxwarn", str(allowed_warnings)]

    box = [str(edge) for edge in box_edges_nm(ligand_gro, box_type, box_distance_nm, min_box_nm)]
    genion = [gmx, "genion", "-s", "ions.tpr", "-o", "ligand_ions.gro", "-p", "system.top",
              "-pname", pname, "-nname", nname, "-neutral"] + (["-conc", str(ion_conc)] if ion_conc > 0 else [])
    commands: dict[Step, list[str]] = {
        "box": [gmx, "editconf", "-f", str(ligand_gro.resolve()), "-o", "ligand_box.gro",
                "-c", "-box", *box, "-bt", box_type],
        "solvate": [gmx, "solvate", "-cp", "ligand_box.gro", "-cs", water_model, "-o", "ligand_solv.gro",
                    "-p", "system.top"],
        "ions_tpr": grompp("ions.mdp", ["ligand_solv.gro"], "ions.tpr", allowed_warnings=1),
        "genion": genion,
        "em_tpr": grompp("em.mdp", ["ligand_ions.gro"], "em.tpr"),
        "em_run": [gmx, "mdrun", "-deffnm", "em", "-v"],
        "nvt_tpr": grompp("nvt.mdp", ["em.gro"], "nvt.tpr"),
        "nvt_run": [gmx, "mdrun", "-deffnm", "nvt", "-v"],
        "npt_tpr": grompp("npt.mdp", ["nvt.gro", "-t", "nvt.cpt"], "npt.tpr"),
        "npt_run": [gmx, "mdrun", "-deffnm", "npt", "-v"],
    }
    for step in STEPS[STEPS.index(start_from):]:
        run_step(step, commands[step], cwd=outdir, stdin="SOL" if step == "genion" else None)

    logger.info("done. NPT output: %s; next: grompp+mdrun production with %s, "
                "then hand the mdrun PID to oneopes monitor", outdir / "npt.gro", outdir / "prod.mdp")
