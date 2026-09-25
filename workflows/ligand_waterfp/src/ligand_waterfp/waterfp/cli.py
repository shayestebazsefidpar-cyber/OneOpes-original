"""
Command-line interface for the WaterFP stage, as one command with three
subcommands (the math itself lives in fingerprint.py, which stays an
importable library module):

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
from collections.abc import Sequence

import MDAnalysis as mda
import numpy as np
import pandas as pd

from ligand_waterfp.selections import select_heavy_atoms, select_water_oxygens
from ligand_waterfp.waterfp.fingerprint import (
    NORM_TAIL_BINS_DEFAULT,
    RDF_BINWIDTH_NM_DEFAULT,
    RDF_RMAX_NM_DEFAULT,
    compute_density_profile,
    fp_from_density_profile,
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


def _compute_density_profiles(args: argparse.Namespace):
    """Shared trajectory pass for `rdf` and `run`: returns
    (atom_names, centers_nm, n_r, end_frame)."""
    u = mda.Universe(args.tpr, args.xtc)
    solute = select_heavy_atoms(u, args.ligand_resname, args.tpr)
    water = select_water_oxygens(u, args.tpr, args.water_resname, args.water_atom_name)

    end_frame = args.end_frame if args.end_frame is not None else len(u.trajectory)
    _, centers_nm, edges_a, shell_vol_nm3 = make_bins(args.rmax_nm, args.binwidth_nm)
    n_r = compute_density_profile(
        solute, water, args.start_frame, end_frame, edges_a, shell_vol_nm3
    )
    return list(solute.names), centers_nm, n_r, end_frame


def _profiles_to_table(
    atom_names: Sequence[str], centers_nm: np.ndarray, n_r: np.ndarray
) -> pd.DataFrame:
    """The long-format rdf.csv table: one row per atom per radial bin
    (columns atom,r_nm,n_r)."""
    atom_names = list(atom_names)
    return pd.DataFrame({
        "atom": np.repeat(atom_names, len(centers_nm)),
        "r_nm": np.tile(centers_nm, len(atom_names)),
        "n_r": np.asarray(n_r).ravel(),
    })


def _fingerprints_from_rdf_table(
    rdf_df: pd.DataFrame, norm_tail_bins: int
) -> pd.DataFrame:
    """Parse an rdf.csv table (columns atom,r_nm,n_r, possibly extras)
    into per-atom arrays and compute each atom's FP (columns atom,fp)."""

    def atom_fp(profile: pd.DataFrame) -> float:
        profile = profile.sort_values("r_nm")
        return fp_from_density_profile(
            profile["n_r"].to_numpy(), profile["r_nm"].to_numpy(), norm_tail_bins
        ).fp

    return pd.DataFrame(
        [
            {"atom": atom, "fp": atom_fp(profile)}
            for atom, profile in rdf_df.groupby("atom", sort=False)
        ]
    )


def cmd_rdf(args: argparse.Namespace) -> None:
    atom_names, centers_nm, n_r, end_frame = _compute_density_profiles(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _profiles_to_table(atom_names, centers_nm, n_r).to_csv(args.out, index=False)
    print(
        f"[waterfp rdf] wrote {args.out} "
        f"({len(atom_names)} atoms x {len(centers_nm)} bins, "
        f"frames [{args.start_frame},{end_frame}))"
    )


def cmd_fingerprint(args: argparse.Namespace) -> None:
    rdf_df = pd.read_csv(args.rdf_csv)
    fp_df = _fingerprints_from_rdf_table(rdf_df, args.norm_tail_bins)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fp_df.to_csv(args.out, index=False)
    print(f"[waterfp fingerprint] wrote {args.out} ({len(fp_df)} atoms)")


def cmd_run(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)

    atom_names, centers_nm, n_r, end_frame = _compute_density_profiles(args)
    rdf_csv = os.path.join(args.outdir, "rdf.csv")
    _profiles_to_table(atom_names, centers_nm, n_r).to_csv(rdf_csv, index=False)

    fp_df = pd.DataFrame({
        "atom": atom_names,
        "fp": [
            fp_from_density_profile(profile, centers_nm, args.norm_tail_bins).fp
            for profile in n_r
        ],
    })
    fp_csv = os.path.join(args.outdir, "fingerprints.csv")
    fp_df.to_csv(fp_csv, index=False)

    print(
        f"[waterfp run] {len(atom_names)} ligand heavy atoms, "
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
    p_fp.add_argument("--norm-tail-bins", type=int, default=NORM_TAIL_BINS_DEFAULT)
    p_fp.add_argument("--out", required=True, help="Output CSV path (columns: atom,fp)")
    p_fp.set_defaults(func=cmd_fingerprint)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
