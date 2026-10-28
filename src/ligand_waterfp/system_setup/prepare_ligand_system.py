"""
Build a standalone ligand-in-water GROMACS system (no protein present):
box definition -> solvation -> ionization -> EM -> NVT -> NPT, ready for
the unbiased production run that the `convergence` subpackage monitors.

Wraps the `gmx` command sequence documented in this folder's README as a
single reproducible script instead of manually-run commands. Takes an
already-prepared ligand structure + topology as input (however you built
those - CGenFF/CHARMM-GUI, GAFF/acpype, extraction from an existing
system's own topology - is outside this workflow's scope, see the README).

Does not run production MD itself - that is left to you (or a thin wrapper
of your own) so you can launch
`ligand_waterfp.convergence.monitor_convergence` against the running
mdrun's PID for online convergence monitoring.

Usage (after `pip install -e .` from the package root):
    ligand-waterfp-prepare-system --ligand-gro ligand.gro --ligand-top ligand.top \\
        --outdir build/ [--box-distance-nm 1.2] [--water-model spc216] \\
        [--ion-conc 0.0] [--mdp-dir mdp]
"""
import argparse
import os
import shutil
import subprocess
import sys

STEPS = [
    "box", "solvate", "ions_tpr", "genion",
    "em_tpr", "em_run", "nvt_tpr", "nvt_run", "npt_tpr", "npt_run",
]


def run(cmd, cwd):
    print(f"[prepare_ligand_system] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ligand-gro", required=True, help="Prepared ligand structure (no water/protein)")
    p.add_argument("--ligand-top", required=True, help="Matching GROMACS topology for --ligand-gro")
    p.add_argument("--outdir", required=True, help="Working directory for all intermediate/output files")
    p.add_argument("--mdp-dir", default=os.path.join(os.path.dirname(__file__), "mdp"),
                    help="Directory containing em.mdp, ions.mdp, nvt.mdp, npt.mdp "
                         "(default: the mdp/ folder next to this script)")
    p.add_argument("--box-distance-nm", type=float, default=1.2)
    p.add_argument("--box-type", default="cubic")
    p.add_argument("--water-model", default="spc216",
                    help="Coordinate file passed to `gmx solvate -cs` (default: spc216, "
                         "GROMACS's bundled 3-point water box - suitable as a solvent box "
                         "template for TIP3P-class water models)")
    p.add_argument("--ion-conc", type=float, default=0.0,
                    help="Ion concentration in mol/L for genion, in addition to neutralization (default: 0.0)")
    p.add_argument("--pname", default="NA")
    p.add_argument("--nname", default="CL")
    p.add_argument("--gmx", default="gmx", help="gmx executable/binary name (default: gmx)")
    p.add_argument("--start-from", choices=STEPS, default=STEPS[0],
                    help="Resume from a specific step instead of the beginning")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ligand_gro = os.path.abspath(args.ligand_gro)
    top = os.path.join(args.outdir, "system.top")
    if not os.path.exists(top):
        shutil.copy(os.path.abspath(args.ligand_top), top)

    def mdp(name):
        path = os.path.join(args.mdp_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"Required mdp file not found: {path}")
        return path

    steps_to_run = STEPS[STEPS.index(args.start_from):]
    gmx = args.gmx

    if "box" in steps_to_run:
        run([gmx, "editconf", "-f", ligand_gro, "-o", "ligand_box.gro",
             "-c", "-d", str(args.box_distance_nm), "-bt", args.box_type], cwd=args.outdir)

    if "solvate" in steps_to_run:
        run([gmx, "solvate", "-cp", "ligand_box.gro", "-cs", args.water_model,
             "-o", "ligand_solv.gro", "-p", "system.top"], cwd=args.outdir)

    if "ions_tpr" in steps_to_run:
        run([gmx, "grompp", "-f", mdp("ions.mdp"), "-c", "ligand_solv.gro",
             "-p", "system.top", "-o", "ions.tpr", "-maxwarn", "1"], cwd=args.outdir)

    if "genion" in steps_to_run:
        genion_cmd = [gmx, "genion", "-s", "ions.tpr", "-o", "ligand_ions.gro",
                      "-p", "system.top", "-pname", args.pname, "-nname", args.nname, "-neutral"]
        if args.ion_conc > 0:
            genion_cmd += ["-conc", str(args.ion_conc)]
        print(f"[prepare_ligand_system] $ echo SOL | {' '.join(genion_cmd)}")
        proc = subprocess.run(genion_cmd, cwd=args.outdir, input="SOL\n", text=True, check=True)

    if "em_tpr" in steps_to_run:
        run([gmx, "grompp", "-f", mdp("em.mdp"), "-c", "ligand_ions.gro",
             "-p", "system.top", "-o", "em.tpr", "-maxwarn", "1"], cwd=args.outdir)
    if "em_run" in steps_to_run:
        run([gmx, "mdrun", "-deffnm", "em", "-v"], cwd=args.outdir)

    if "nvt_tpr" in steps_to_run:
        run([gmx, "grompp", "-f", mdp("nvt.mdp"), "-c", "em.gro",
             "-p", "system.top", "-o", "nvt.tpr", "-maxwarn", "1"], cwd=args.outdir)
    if "nvt_run" in steps_to_run:
        run([gmx, "mdrun", "-deffnm", "nvt", "-v"], cwd=args.outdir)

    if "npt_tpr" in steps_to_run:
        run([gmx, "grompp", "-f", mdp("npt.mdp"), "-c", "nvt.gro", "-t", "nvt.cpt",
             "-p", "system.top", "-o", "npt.tpr", "-maxwarn", "1"], cwd=args.outdir)
    if "npt_run" in steps_to_run:
        run([gmx, "mdrun", "-deffnm", "npt", "-v"], cwd=args.outdir)

    print(f"[prepare_ligand_system] done. NPT output: {os.path.join(args.outdir, 'npt.gro')}")
    print("[prepare_ligand_system] next: grompp+mdrun production with mdp/prod.mdp, then hand the "
          "mdrun PID to ligand-waterfp-monitor")


if __name__ == "__main__":
    main()
