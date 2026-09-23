"""
Build the ranking CSV that run_official_selection.py expects
(columns: name,atom,fp,fp_round) from the waterfp subpackage's output
(fingerprints.csv: atom,fp).

This closes the one adaptation step the official WaterFP notebook leaves
entirely up to the user: it never publishes how its own `ranking.csv` was
built from raw FP values, only how it's consumed
(`c[c['name']==mol].sort_values('fp')` inside `ranking()` /
`data_for_ranking()`, see run_official_selection.py). The convention this
script follows - atom serial = 1-based position of that atom in
`Universe(tpr).select_atoms(f"resname {ligand_resname} and not name H*")`
- is chosen specifically because it is the SAME selection expression
`load_mol()` uses internally, so the serials line up with what
`select_next_atom`/`select_bulk_atom` index into. This is a necessary,
system-independent piece of glue code, not part of the official algorithm
itself.

Usage:
    ligand-waterfp-prepare-ranking-csv --tpr ligand.tpr --fp-csv outputs/fingerprints.csv \\
        --system-id my_ligand --out outputs/selection/ranking_input.csv \\
        [--ligand-resname MOL] [--fp-round-decimals 1]
"""
import argparse
import os
import pandas as pd
import MDAnalysis as mda


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpr", required=True,
                   help="Same topology load_mol() will open as '<--tpr-prefix>.tpr' in run_official_selection.py")
    p.add_argument("--fp-csv", required=True, help="the waterfp subpackage's fingerprints.csv (columns: atom,fp)")
    p.add_argument("--system-id", required=True, help="Value to store in the 'name' column")
    p.add_argument("--ligand-resname", default="MOL")
    p.add_argument("--fp-round-decimals", type=int, default=1,
                   help="fp_round = round(fp, N) (default: 1) - see module docstring for why this "
                        "default is a reasonable placeholder, not a confirmed upstream convention")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    u = mda.Universe(args.tpr)
    heavy = u.select_atoms(f"resname {args.ligand_resname} and not name H*")
    if len(heavy) == 0:
        raise SystemExit(f"No heavy atoms found for resname '{args.ligand_resname}' in {args.tpr}")
    # 1-based serial = position in this exact selection, matching load_mol()'s own convention
    serial_by_name = {atom.name: i + 1 for i, atom in enumerate(heavy)}

    fp_df = pd.read_csv(args.fp_csv)
    missing = set(fp_df["atom"]) - set(serial_by_name)
    if missing:
        raise SystemExit(f"Atoms in {args.fp_csv} not found in {args.tpr}'s "
                          f"resname={args.ligand_resname} heavy-atom selection: {sorted(missing)}")

    out_df = pd.DataFrame({
        "name": args.system_id,
        "atom": fp_df["atom"].map(serial_by_name),
        "fp": fp_df["fp"],
    })
    out_df["fp_round"] = out_df["fp"].round(args.fp_round_decimals)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[prepare_ranking_csv] {len(out_df)} atoms, wrote {args.out}")


if __name__ == "__main__":
    main()
