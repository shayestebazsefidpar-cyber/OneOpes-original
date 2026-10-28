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

Both are computed by the shared `waterfp.WaterFP` analysis. Writes two CSVs to --outdir:
    rdf.csv           (atom,r_nm,n_r)          - water-O number density n(r), waters/nm^3
    fingerprints.csv  (atom,fp)                - WaterFP value per ligand heavy atom

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

from waterfp import NORM_TAIL_BINS_DEFAULT, RDF_BINWIDTH_NM_DEFAULT, RDF_RMAX_NM_DEFAULT, WaterFP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpr", required=True)
    p.add_argument("--xtc", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--ligand-resname", default="MOL")
    p.add_argument("--water-resname", default="SOL")
    p.add_argument("--water-atom-name", default="O")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--rmax-nm", type=float, default=RDF_RMAX_NM_DEFAULT)
    p.add_argument("--binwidth-nm", type=float, default=RDF_BINWIDTH_NM_DEFAULT)
    p.add_argument("--norm-tail-bins", type=int, default=NORM_TAIL_BINS_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    u = mda.Universe(args.tpr, args.xtc)
    solute = u.select_atoms(f"resname {args.ligand_resname} and not name H*")
    water = u.select_atoms(f"resname {args.water_resname} and name {args.water_atom_name}")
    if len(solute) == 0:
        raise SystemExit(f"No heavy atoms found for resname '{args.ligand_resname}' in {args.tpr}")
    if len(water) == 0:
        raise SystemExit(f"No water oxygens found for resname '{args.water_resname}' / "
                          f"atom name '{args.water_atom_name}' in {args.tpr}")

    end_frame = args.end_frame if args.end_frame is not None else len(u.trajectory)
    waterfp = WaterFP(solute, water, rmax_nm=args.rmax_nm, binwidth_nm=args.binwidth_nm,
                      norm_tail_bins=args.norm_tail_bins)
    waterfp.run(start=args.start_frame, stop=end_frame)

    rdf_csv = os.path.join(args.outdir, "rdf.csv")
    waterfp.profile_table()[["atom", "r_nm", "n_r"]].to_csv(rdf_csv, index=False)

    fp_csv = os.path.join(args.outdir, "fingerprints.csv")
    # fingerprints.csv keeps its lowercase "fp" column: prepare_ranking_csv reads it by that name
    waterfp.fp_table().rename(columns={"FP": "fp"}).to_csv(fp_csv, index=False)

    print(f"[run_waterfp] {len(solute)} ligand heavy atoms, frames [{args.start_frame},{end_frame})")
    print(f"[run_waterfp] wrote {rdf_csv}")
    print(f"[run_waterfp] wrote {fp_csv}")


if __name__ == "__main__":
    main()
