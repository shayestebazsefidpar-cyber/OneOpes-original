"""
Command-line interface for the WaterFP stage, as one command with three
subcommands (the math itself lives in calculate_rdf.py and
calculate_fingerprint.py, which stay importable library modules):

    ligand-waterfp run          - trajectory -> rdf.csv + fingerprints.csv
                                  (the single-shot stage the workflow uses)
    ligand-waterfp rdf          - trajectory -> rdf.csv only (the expensive
                                  trajectory pass, cacheable)
    ligand-waterfp fingerprint  - rdf.csv -> fingerprints.csv (cheap;
                                  re-fingerprint a cached rdf.csv, e.g.
                                  with a different --norm-tail-bins)

Restrict --start-frame/--end-frame to the convergence stage's converged
block - see ../convergence/README.md's "Next stage" section.
"""

import argparse
import os

import MDAnalysis as mda
import pandas as pd

from ligand_waterfp.selections import select_heavy_atoms, select_water_oxygens
from ligand_waterfp.waterfp.calculate_fingerprint import (
    NORM_TAIL_BINS_DEFAULT,
    fingerprints_from_rdf_table,
)
from ligand_waterfp.waterfp.calculate_rdf import (
    RDF_BINWIDTH_NM_DEFAULT,
    RDF_RMAX_NM_DEFAULT,
    compute_density_profile,
    make_bins,
)


def _add_trajectory_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tpr", required=True)
    p.add_argument("--xtc", required=True)
    p.add_argument("--ligand-resname", default="MOL")
    p.add_argument(
        "--water-resname",
        default=None,
        help="Default: auto-detect via MDAnalysis's 'water' selection keyword",
    )
    p.add_argument(
        "--water-atom-name",
        default=None,
        help="Default: atoms named O* within the water residues",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Default: last frame of the trajectory",
    )
    p.add_argument("--rmax-nm", type=float, default=RDF_RMAX_NM_DEFAULT)
    p.add_argument("--binwidth-nm", type=float, default=RDF_BINWIDTH_NM_DEFAULT)


def _compute_rdf_table(args: argparse.Namespace):
    """Shared trajectory pass for `rdf` and `run`: returns the long-format
    RDF table (columns atom,r_nm,n_r) plus (n_atoms, n_bins, end_frame)
    for the summary lines."""
    u = mda.Universe(args.tpr, args.xtc)
    solute = select_heavy_atoms(u, args.ligand_resname, args.tpr)
    water = select_water_oxygens(u, args.tpr, args.water_resname, args.water_atom_name)

    end_frame = args.end_frame if args.end_frame is not None else len(u.trajectory)
    _, centers_nm, edges_a, shell_vol_nm3 = make_bins(args.rmax_nm, args.binwidth_nm)
    n_r = compute_density_profile(
        solute, water, args.start_frame, end_frame, edges_a, shell_vol_nm3
    )

    rows = []
    for name, profile in zip(solute.names, n_r):
        for r_nm, val in zip(centers_nm, profile):
            rows.append({"atom": name, "r_nm": r_nm, "n_r": val})
    return pd.DataFrame(rows), len(solute), len(centers_nm), end_frame


def cmd_rdf(args: argparse.Namespace) -> None:
    rdf_df, n_atoms, n_bins, end_frame = _compute_rdf_table(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rdf_df.to_csv(args.out, index=False)
    print(
        f"[waterfp rdf] wrote {args.out} "
        f"({n_atoms} atoms x {n_bins} bins, frames [{args.start_frame},{end_frame}))"
    )


def cmd_fingerprint(args: argparse.Namespace) -> None:
    rdf_df = pd.read_csv(args.rdf_csv)
    fp_df = fingerprints_from_rdf_table(rdf_df, args.binwidth_nm, args.norm_tail_bins)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fp_df.to_csv(args.out, index=False)
    print(f"[waterfp fingerprint] wrote {args.out} ({len(fp_df)} atoms)")


def cmd_run(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)

    rdf_df, n_atoms, _, end_frame = _compute_rdf_table(args)
    rdf_csv = os.path.join(args.outdir, "rdf.csv")
    rdf_df.to_csv(rdf_csv, index=False)

    fp_df = fingerprints_from_rdf_table(rdf_df, args.binwidth_nm, args.norm_tail_bins)
    fp_csv = os.path.join(args.outdir, "fingerprints.csv")
    fp_df.to_csv(fp_csv, index=False)

    print(
        f"[waterfp run] {n_atoms} ligand heavy atoms, "
        f"frames [{args.start_frame},{end_frame})"
    )
    print(f"[waterfp run] wrote {rdf_csv}")
    print(f"[waterfp run] wrote {fp_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ligand-waterfp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="trajectory -> rdf.csv + fingerprints.csv in --outdir"
    )
    _add_trajectory_args(p_run)
    p_run.add_argument("--norm-tail-bins", type=int, default=NORM_TAIL_BINS_DEFAULT)
    p_run.add_argument("--outdir", required=True)
    p_run.set_defaults(func=cmd_run)

    p_rdf = sub.add_parser("rdf", help="trajectory -> rdf.csv only")
    _add_trajectory_args(p_rdf)
    p_rdf.add_argument(
        "--out", required=True, help="Output CSV path (columns: atom,r_nm,n_r)"
    )
    p_rdf.set_defaults(func=cmd_rdf)

    p_fp = sub.add_parser(
        "fingerprint", help="rdf.csv -> fingerprints.csv (no trajectory needed)"
    )
    p_fp.add_argument(
        "--rdf-csv",
        required=True,
        help="Output of the rdf/run subcommand (columns: atom,r_nm,n_r)",
    )
    p_fp.add_argument(
        "--binwidth-nm",
        type=float,
        default=RDF_BINWIDTH_NM_DEFAULT,
        help="Must match the bin width used to produce --rdf-csv "
        f"(default: {RDF_BINWIDTH_NM_DEFAULT})",
    )
    p_fp.add_argument("--norm-tail-bins", type=int, default=NORM_TAIL_BINS_DEFAULT)
    p_fp.add_argument("--out", required=True, help="Output CSV path (columns: atom,fp)")
    p_fp.set_defaults(func=cmd_fingerprint)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
