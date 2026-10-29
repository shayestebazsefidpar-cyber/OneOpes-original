"""Build and equilibrate a ligand-in-water GROMACS system (``oneopes prepare-system``)."""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, get_args

from cyclopts.types import ExistingFile

Step = Literal["box", "solvate", "ions_tpr", "genion", "em_tpr", "em_run", "nvt_tpr", "nvt_run", "npt_tpr", "npt_run"]
STEPS = get_args(Step)
MDP_DIR = Path(__file__).parent / "mdp"


def run_command(cmd, cwd):
    print(f"[prepare_ligand_system] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_system(*, ligand_gro: ExistingFile, ligand_top: ExistingFile, outdir: Path, mdp_dir: Path = MDP_DIR,
                   box_distance_nm: float = 1.2, box_type: str = "cubic", water_model: str = "spc216",
                   ion_conc: float = 0.0, pname: str = "NA", nname: str = "CL", gmx: str = "gmx",
                   start_from: Step = "box") -> None:
    """Build and equilibrate a ligand-in-water GROMACS system for the unbiased run the monitor follows.

    Box definition -> solvation -> ionization -> EM -> NVT -> NPT, with no protein present
    (see docs/ligand_setup.md). Production MD is left to you, so that `oneopes monitor` can
    follow the running mdrun's PID.

    Parameters
    ----------
    ligand_gro
        Prepared ligand structure (no water or protein).
    ligand_top
        Matching GROMACS topology.
    outdir
        Working directory for all intermediate and output files.
    mdp_dir
        Directory with em.mdp, ions.mdp, nvt.mdp and npt.mdp.
    water_model
        Coordinate file for `gmx solvate -cs` (spc216 suits TIP3P-class water models).
    ion_conc
        Ion concentration (mol/L) for genion, on top of neutralization.
    gmx
        gmx executable.
    start_from
        Resume from this step instead of the beginning.
    """
    os.makedirs(outdir, exist_ok=True)

    ligand_gro = os.path.abspath(ligand_gro)
    top = os.path.join(outdir, "system.top")
    if not os.path.exists(top):
        shutil.copy(os.path.abspath(ligand_top), top)

    def mdp(name):
        path = os.path.join(mdp_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"Required mdp file not found: {path}")
        return path

    steps_to_run = STEPS[STEPS.index(start_from):]

    if "box" in steps_to_run:
        run_command([gmx, "editconf", "-f", ligand_gro, "-o", "ligand_box.gro",
             "-c", "-d", str(box_distance_nm), "-bt", box_type], cwd=outdir)

    if "solvate" in steps_to_run:
        run_command([gmx, "solvate", "-cp", "ligand_box.gro", "-cs", water_model,
             "-o", "ligand_solv.gro", "-p", "system.top"], cwd=outdir)

    if "ions_tpr" in steps_to_run:
        run_command([gmx, "grompp", "-f", mdp("ions.mdp"), "-c", "ligand_solv.gro",
             "-p", "system.top", "-o", "ions.tpr", "-maxwarn", "1"], cwd=outdir)

    if "genion" in steps_to_run:
        genion_cmd = [gmx, "genion", "-s", "ions.tpr", "-o", "ligand_ions.gro",
                      "-p", "system.top", "-pname", pname, "-nname", nname, "-neutral"]
        if ion_conc > 0:
            genion_cmd += ["-conc", str(ion_conc)]
        print(f"[prepare_ligand_system] $ echo SOL | {' '.join(genion_cmd)}")
        subprocess.run(genion_cmd, cwd=outdir, input="SOL\n", text=True, check=True)

    if "em_tpr" in steps_to_run:
        run_command([gmx, "grompp", "-f", mdp("em.mdp"), "-c", "ligand_ions.gro",
             "-p", "system.top", "-o", "em.tpr", "-maxwarn", "1"], cwd=outdir)
    if "em_run" in steps_to_run:
        run_command([gmx, "mdrun", "-deffnm", "em", "-v"], cwd=outdir)

    if "nvt_tpr" in steps_to_run:
        run_command([gmx, "grompp", "-f", mdp("nvt.mdp"), "-c", "em.gro",
             "-p", "system.top", "-o", "nvt.tpr", "-maxwarn", "1"], cwd=outdir)
    if "nvt_run" in steps_to_run:
        run_command([gmx, "mdrun", "-deffnm", "nvt", "-v"], cwd=outdir)

    if "npt_tpr" in steps_to_run:
        run_command([gmx, "grompp", "-f", mdp("npt.mdp"), "-c", "nvt.gro", "-t", "nvt.cpt",
             "-p", "system.top", "-o", "npt.tpr", "-maxwarn", "1"], cwd=outdir)
    if "npt_run" in steps_to_run:
        run_command([gmx, "mdrun", "-deffnm", "npt", "-v"], cwd=outdir)

    print(f"[prepare_ligand_system] done. NPT output: {os.path.join(outdir, 'npt.gro')}")
    print("[prepare_ligand_system] next: grompp+mdrun production with mdp/prod.mdp, then hand the "
          "mdrun PID to oneopes monitor")
