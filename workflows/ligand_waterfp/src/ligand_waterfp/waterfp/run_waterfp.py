"""
End-to-end WaterFP calculation over a (typically already-converged, per
the sibling `convergence` subpackage) ligand-in-water trajectory: RDF for
every ligand heavy atom, then the WaterFP fingerprint derived from it.

This is the single-shot counterpart to
`ligand_waterfp.convergence.monitor_convergence`, which repeats the same
two calculations block-by-block to test for convergence rather than
computing a single final answer. Use this script once you already trust
the trajectory is long enough (e.g. after monitor_convergence has signaled
convergence).

Writes two CSVs to --outdir:
    rdf.csv           (atom,r_nm,n_r)          - calculate_rdf.py's format
    fingerprints.csv  (atom,fp)                - calculate_fingerprint.py's format

fingerprints.csv is the input official_selection.prepare_ranking_csv
consumes to build the ranking CSV official_selection.run_official_selection
needs - see that subpackage's README.

Usage (after `pip install -e .` from the package root):
    ligand-waterfp-run-waterfp --tpr prod.tpr --xtc prod.xtc --outdir outputs/fingerprints \\
        [--ligand-resname MOL] [--water-resname SOL] [--water-atom-name O] \\
        [--start-frame 0] [--end-frame N]
"""

import argparse
import os

import MDAnalysis as mda
import pandas as pd

from ligand_waterfp.selections import select_heavy_atoms, select_water_oxygens

from .calculate_fingerprint import fingerprints_from_rdf_table
from .calculate_rdf import compute_density_profile, make_bins


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tpr", required=True)
    p.add_argument("--xtc", required=True)
    p.add_argument("--outdir", required=True)
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
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--rmax-nm", type=float, default=2.001)
    p.add_argument("--binwidth-nm", type=float, default=0.001)
    p.add_argument("--norm-tail-bins", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    u = mda.Universe(args.tpr, args.xtc)
    solute = select_heavy_atoms(u, args.ligand_resname, args.tpr)
    water = select_water_oxygens(u, args.tpr, args.water_resname, args.water_atom_name)

    end_frame = args.end_frame if args.end_frame is not None else len(u.trajectory)
    edges_nm, centers_nm, edges_a, shell_vol_nm3 = make_bins(
        args.rmax_nm, args.binwidth_nm
    )
    n_r = compute_density_profile(
        solute, water, args.start_frame, end_frame, edges_a, shell_vol_nm3
    )

    rdf_rows = []
    for ai, name in enumerate(solute.names):
        for r_nm, val in zip(centers_nm, n_r[ai]):
            rdf_rows.append({"atom": name, "r_nm": r_nm, "n_r": val})
    rdf_df = pd.DataFrame(rdf_rows)
    rdf_csv = os.path.join(args.outdir, "rdf.csv")
    rdf_df.to_csv(rdf_csv, index=False)

    fp_df = fingerprints_from_rdf_table(rdf_df, args.binwidth_nm, args.norm_tail_bins)
    fp_csv = os.path.join(args.outdir, "fingerprints.csv")
    fp_df.to_csv(fp_csv, index=False)

    print(
        f"[run_waterfp] {len(solute)} ligand heavy atoms, frames [{args.start_frame},{end_frame})"
    )
    print(f"[run_waterfp] wrote {rdf_csv}")
    print(f"[run_waterfp] wrote {fp_csv}")


if __name__ == "__main__":
    main()
