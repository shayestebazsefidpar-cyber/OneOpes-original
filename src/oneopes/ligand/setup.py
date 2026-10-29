"""Build and equilibrate a ligand-in-water GROMACS system (``oneopes prepare-system``)."""
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal, get_args

from cyclopts.types import ExistingFile

from oneopes.errors import InputError

logger = logging.getLogger(__name__)

Step = Literal["box", "solvate", "ions_tpr", "genion", "em_tpr", "em_run", "nvt_tpr", "nvt_run", "npt_tpr", "npt_run"]
STEPS = get_args(Step)
MDP_DIR = Path(__file__).parent / "mdp"


def run_command(cmd: list[str], cwd: Path, stdin: str | None = None) -> None:
    """Run ``cmd`` in ``cwd``, logging it first; ``stdin`` answers its prompt (as ``echo stdin | cmd``)."""
    shown = shlex.join(cmd) if stdin is None else f"echo {stdin} | {shlex.join(cmd)}"
    logger.info("$ %s", shown)
    subprocess.run(cmd, cwd=cwd, check=True, input=None if stdin is None else f"{stdin}\n", text=True)


def prepare_system(*, ligand_gro: ExistingFile, ligand_top: ExistingFile, outdir: Path, mdp_dir: Path = MDP_DIR,
                   box_distance_nm: float = 1.2, box_type: str = "cubic", water_model: str = "spc216",
                   ion_conc: float = 0.0, pname: str = "NA", nname: str = "CL", gmx: str = "gmx",
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
    water_model
        Coordinate file for `gmx solvate -cs` (spc216 suits TIP3P-class water models).
    ion_conc
        Ion concentration (mol/L) for genion, on top of neutralization.
    gmx
        gmx executable.
    start_from
        Resume from this step instead of the beginning.
    """
    for name in ("ions.mdp", "em.mdp", "nvt.mdp", "npt.mdp", "prod.mdp"):
        if not (mdp_dir / name).is_file():
            raise InputError(f"Required mdp file not found: {mdp_dir / name}")
    outdir.mkdir(parents=True, exist_ok=True)
    if not (outdir / "system.top").exists():
        shutil.copy(ligand_top, outdir / "system.top")
    if not (outdir / "prod.mdp").exists():
        shutil.copy(mdp_dir / "prod.mdp", outdir / "prod.mdp")

    def grompp(mdp: str, structure: list[str], tpr: str) -> list[str]:
        return [gmx, "grompp", "-f", str(mdp_dir / mdp), "-c", *structure, "-p", "system.top", "-o", tpr,
                "-maxwarn", "1"]

    genion = [gmx, "genion", "-s", "ions.tpr", "-o", "ligand_ions.gro", "-p", "system.top",
              "-pname", pname, "-nname", nname, "-neutral"] + (["-conc", str(ion_conc)] if ion_conc > 0 else [])
    commands: dict[Step, list[str]] = {
        "box": [gmx, "editconf", "-f", str(ligand_gro.resolve()), "-o", "ligand_box.gro",
                "-c", "-d", str(box_distance_nm), "-bt", box_type],
        "solvate": [gmx, "solvate", "-cp", "ligand_box.gro", "-cs", water_model, "-o", "ligand_solv.gro",
                    "-p", "system.top"],
        "ions_tpr": grompp("ions.mdp", ["ligand_solv.gro"], "ions.tpr"),
        "genion": genion,
        "em_tpr": grompp("em.mdp", ["ligand_ions.gro"], "em.tpr"),
        "em_run": [gmx, "mdrun", "-deffnm", "em", "-v"],
        "nvt_tpr": grompp("nvt.mdp", ["em.gro"], "nvt.tpr"),
        "nvt_run": [gmx, "mdrun", "-deffnm", "nvt", "-v"],
        "npt_tpr": grompp("npt.mdp", ["nvt.gro", "-t", "nvt.cpt"], "npt.tpr"),
        "npt_run": [gmx, "mdrun", "-deffnm", "npt", "-v"],
    }
    for step in STEPS[STEPS.index(start_from):]:
        run_command(commands[step], cwd=outdir, stdin="SOL" if step == "genion" else None)

    logger.info("done. NPT output: %s; next: grompp+mdrun production with %s, "
                "then hand the mdrun PID to oneopes monitor", outdir / "npt.gro", outdir / "prod.mdp")
