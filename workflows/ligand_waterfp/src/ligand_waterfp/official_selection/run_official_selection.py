"""
Driver for the OFFICIAL WaterFP atom-selection algorithm.

Upstream source: https://github.com/valeriorizzi/WaterFP
                  Scripts/fp_driven_atom_selection.ipynb, cells 2 and 5
Vendored commit:  6a5aef725a66b28e86d3582aee3e9e8bbbe4bc54
                  (see WaterFP_official_scripts/README.md)

Everything between the BEGIN/END OFFICIAL CODE markers is copied verbatim
from the notebook above: `load_mol()`, `path_length()`, `ranking()`,
`select_next_atom()`, `select_bulk_atom()` - same signatures, same logic,
same variable names and control flow, cell order preserved. Only
whitespace/PEP8 formatting (blank lines, spacing around `:`/`==`, trailing
spaces) was normalized when copying out of the notebook's JSON source -
diffed directly against the notebook's own cells 2 and 5 to confirm no
logic difference beyond that. All five functions are reproduced -
`ranking()` is not optional supporting code, it is what `select_next_atom`/
`select_bulk_atom` call internally to get the FP-sorted table, exactly as
written upstream (referencing a module-level `c` DataFrame, the same
pattern the notebook itself uses via its own global `c`). Nothing in this
block is rewritten, simplified, or reinterpreted - the official selection
logic must always be traced back to this file, never reimplemented
elsewhere in this workflow.

Only `main()`, below the marked block, is new: it parses CLI args, builds
the module-level `c` DataFrame from a system-independent ranking CSV
(instead of the authors' own hardcoded ranking.csv path), calls the
verbatim functions - mirroring exactly what running the notebook's cells 0,
1, 2, 4, 5, 6 in order would do, with cell 1's hardcoded path replaced by
--ranking-csv - and, if --out is given, parses the two result sentences
and writes the structured G1/G2 YAML directly (see
`ligand_waterfp.g1_g2_selection.select_g1_g2`), all in this same process.
There is no intermediate text file and no separate CLI stage for that
anymore: the sentences never leave this function.

Input CSV format (--ranking-csv), one row per ligand heavy atom - see
`ligand_waterfp.official_selection.prepare_ranking_csv` to build this from
the waterfp subpackage's fingerprints.csv:
    name,atom,fp,fp_round
    <system_id>,<1-based atom serial in the .tpr>,<FP value>,<rounded FP>

`fp_round` groups atoms the official algorithm treats as "the same FP" for
tie-breaking (its own code compares `fp_round` values with a fixed 0.2
tolerance - see `select_next_atom`/`select_bulk_atom` below). The exact
rounding convention used in the authors' own ranking.csv-generation cell
was not part of the two verbatim cells taken from the notebook (cell 1,
which builds their own ranking.csv, was not published in a form we could
vendor), so if your CSV does not already include `fp_round`, this script
fills it in as `round(fp, --fp-round-decimals)` (default 1 decimal place)
- a reasonable default, not a confirmed reproduction of the authors' own
convention. Override with --fp-round-decimals, or precompute your own
`fp_round` column, if this matters for your use case.

Usage:
    ligand-waterfp-select --ranking-csv ranking.csv --system-id myproject-ligandA \\
        [--out outputs/selection/g1_g2.yaml]
        (expects my_ligand.tpr - see load_mol()'s docstring note below)
"""
import argparse
import MDAnalysis
import numpy
import networkx
import pandas
import re

from ligand_waterfp.g1_g2_selection.select_g1_g2 import parse_selection_lines, write_g1_g2_yaml

# ============================= BEGIN OFFICIAL CODE =============================
# verbatim from WaterFP/Scripts/fp_driven_atom_selection.ipynb, cells 2 and 5
# (commit 6a5aef725a66b28e86d3582aee3e9e8bbbe4bc54). Do not edit this block -
# see the module docstring above and WaterFP_official_scripts/README.md.
#
# `c` is a module-level DataFrame that ranking() reads, exactly as in the
# notebook (where cell 1 assigns the module-level `c` before cell 2's
# functions are ever called). It is assigned in main(), below, via
# `global c` - Python resolves `c` inside ranking() at call time, so this
# reproduces the notebook's own cell-execution-order behavior without
# altering ranking()'s body or any function signature.


def load_mol(mol: str):
    # load connectivity information form tpr file
    u = MDAnalysis.Universe(mol+'.tpr')
    # select heavy atoms
    ha = u.select_atoms("resname MOL and not (name H*)")

    # create graph of heavy atoms

    G = networkx.Graph()

    # add heavy atoms as nodes
    for atom in ha:
        G.add_node(atom.index, element=atom.name)  # extract element form atom name: re.search(r'(\D+)', atom.name).group(1)
    for bond in ha.bonds:
        # only add edges between heavy atoms (ignoring hydrogens)
        if bond.atoms[0].index in G.nodes and bond.atoms[1].index in G.nodes:
            G.add_edge(bond.atoms[0].index, bond.atoms[1].index)

    return u, ha, G


def path_length(atom1, atom2, G):
    try:
        length = networkx.shortest_path_length(G, source=atom1.index, target=atom2.index)
    except networkx.NetworkXNoPath:
        print(f"No path between {atom1.index} and {atom2.index}")
        length = None
    return length


def ranking(mol: str):
    data = c[c['name'] == mol].sort_values('fp')
    return data


def select_next_atom(mol: str, verbose=True):

    data = ranking(mol)
    u, ha, G = load_mol(mol.split('-')[0]+mol.split('-')[1])

    """
    Walk through the fp-sorted table and select the anti-bulk fp atoms.

    Rules:
      - First atom is always the first row in the table
      - Reject atoms directly connected to the first selected atom (path = 1)
      - Compare atoms with fp_round difference < 0.2
      - Among atoms with equal fp, pick the atom with the longest path
      - Loop through the entire table, until the fp_round difference is not < 0.2
    """

    # first atom is always the first row in the table
    # zero-index in MDAnalysis, one-index in .gro, used for ranking dataframe

    anchor_idx = int(data.iloc[0]['atom']) - 1
    anchor_atom = ha[anchor_idx]

    if verbose:
        print(f"Highest-ranking atom: {anchor_atom.name} ({int(anchor_atom.index+1)}) with fp_round {data.iloc[0]['fp_round']}")

    # continue with second entry in the table

    i = 1
    winner = None
    l_win = -1

    # check if there are more than 2 atoms with identical fp_round as anchor
    if data[data['fp_round'] == data.iloc[0]['fp_round']].count().values[0] > 2:
        if verbose:
            print(">2 atoms with identical fp_round as highest-ranking atom -> ambiguous choice")
        return None
    while i < len(data):
        cand_idx = int(data.iloc[i]['atom']) - 1
        cand_atom = ha[cand_idx]
        l_cand = path_length(cand_atom, anchor_atom, G)

        if l_cand <= 1:
            if verbose:
                print(f"{i}: {cand_atom.name} connected to {anchor_atom.name} -> rejected")
            i += 1
            continue

        if winner is None:
            winner = cand_atom
            l_win = l_cand
            fp_win = float(data.iloc[i]['fp_round'])
        else:
            fp_cand = float(data.iloc[i]['fp_round'])

            if abs(fp_cand - fp_win) < 0.2:

                # candidate is in close fp group
                if l_cand > l_win:
                    if verbose:
                        print(f"{i}: {cand_atom.name} beats {winner.name} (longer path {l_cand} > {l_win})")
                    winner, l_win, fp_win = cand_atom, l_cand, fp_cand
                elif l_cand == l_win:
                    if verbose:
                        print(f"{i}: {cand_atom.name} and {winner.name} are equidistant and have same fp -> choose either")
                        if abs(float(data.iloc[i]['fp_round'])) < 2.0:
                            print("Warning: Very low fingerprint value!")
                    break
                else:
                    if verbose:
                        print(f"{i}: {cand_atom.name} shorter path than {winner.name} -> ignored")

        i += 1

    return f"anti-bulk fp selection: {anchor_atom.name} ({int(anchor_atom.index+1)}), {winner.name} ({int(winner.index+1)})"


def select_bulk_atom(mol: str, verbose=True):

    data = ranking(mol)
    u, ha, G = load_mol(mol.split('-')[0]+mol.split('-')[1])

    """
    Walk through the fp-sorted table and select the anti-bulk fp atoms.

    Rules:
      - First atom is always the first row in the table
      - Reject atoms directly connected to the first selected atom (path = 1)
      - Compare atoms with fp_round difference < 0.2
      - Among atoms with equal fp, pick the atom with the longest path
      - Loop through the entire table, until the fp_round difference is not < 0.2
    """

    # first atom is always the last row in the table
    # zero-index in MDAnalysis, one-index in .gro, used for ranking dataframe

    anchor_idx = int(data.iloc[-1]['atom']) - 1
    anchor_atom = ha[anchor_idx]

    if verbose:
        print(f"Lowest-ranking atom: {anchor_atom.name} ({int(anchor_atom.index+1)}) with fp_round {data.iloc[-1]['fp_round']}")

    # continue with second entry from the bottom in the table

    i = -2
    winner = None
    l_win = -1
    fp_win = float(data.iloc[i]['fp_round'])

    # check if there are more than 2 atoms with identical fp_round as anchor
    while abs(i) <= len(data):

        cand_idx = int(data.iloc[i]['atom']) - 1
        cand_atom = ha[cand_idx]
        l_cand = path_length(cand_atom, anchor_atom, G)
        fp_cand = float(data.iloc[i]['fp_round'])

        #if abs(fp_cand - fp_win) < 0.2:

        if l_cand <= 2:
            if verbose:
                print(f"{i}: {cand_atom.name} connected to {anchor_atom.name} -> rejected")
        else:
            if winner is None:
                winner = cand_atom
                l_win = l_cand
                break
            else:
                winner = cand_atom
                l_win = l_cand
                break

            """ else:
                if l_cand > l_win:
                    if verbose:
                        print(f"{i}: {cand_atom.name} beats {winner.name} "
                              f"(longer path {l_cand} > {l_win})")
                    winner, l_win = cand_atom, l_cand
                elif l_cand == l_win:
                    if verbose:
                        print(f"{i}: {cand_atom.name} and {winner.name} are equidistant "
                              f"(path {l_cand}) -> choose either")
                else:
                    if verbose:
                        print(f"{i}: {cand_atom.name} shorter path than {winner.name} -> ignored")
 """
        i -= 1  # walk upward

    return f"bulk fp selection: {anchor_atom.name} ({int(anchor_atom.index+1)}), " \
           f"{winner.name} ({int(winner.index+1)})"

# ============================== END OFFICIAL CODE ==============================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranking-csv", required=True,
                   help="CSV with columns name,atom,fp[,fp_round] - see module docstring, "
                        "and `ligand_waterfp.official_selection.prepare_ranking_csv` to build it.")
    p.add_argument("--system-id", required=True,
                   help="Value matched against the CSV's 'name' column. Also determines the "
                        ".tpr filename load_mol() opens, per the official code's own convention: "
                        "mol.split('-')[0]+mol.split('-')[1]+'.tpr' - e.g. system-id 'SystemA-lig1' "
                        "-> 'SystemAlig1.tpr' must exist in the working directory.")
    p.add_argument("--fp-round-decimals", type=int, default=1,
                   help="If the input CSV has no fp_round column, derive it as round(fp, N) (default: 1).")
    p.add_argument("--out", default=None,
                   help="Optional path to write the structured G1/G2 result to, e.g. "
                        "outputs/selection/g1_g2.yaml (see "
                        "ligand_waterfp.g1_g2_selection.select_g1_g2.write_g1_g2_yaml). "
                        "If omitted, the two result sentences are only printed.")
    return p.parse_args()


def main():
    global c
    args = parse_args()

    c = pandas.read_csv(args.ranking_csv, header=[0])
    if "fp_round" not in c.columns:
        c["fp_round"] = c["fp"].round(args.fp_round_decimals)

    mol = args.system_id

    print("=" * 70)
    print("Ranking table (ascending FP, as used by ranking()):")
    print(ranking(mol).to_string(index=False))
    print("=" * 70)
    anti_bulk = select_next_atom(mol, verbose=True)
    print("=" * 70)
    print("select_next_atom (anti-bulk fp pair):")
    print(anti_bulk)
    print("=" * 70)
    bulk = select_bulk_atom(mol, verbose=True)
    print("select_bulk_atom (bulk fp pair):")
    print(bulk)
    print("=" * 70)

    if args.out:
        result = parse_selection_lines(f"{anti_bulk}\n{bulk}\n")
        if "G1" not in result or "G2" not in result:
            raise SystemExit(
                "Could not parse both an anti-bulk and a bulk selection line from the "
                "official algorithm's own output above - --out was not written."
            )
        write_g1_g2_yaml(result, args.out, system_id=args.system_id)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
