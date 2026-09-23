"""
Turn the official WaterFP selection algorithm's raw text output (from
`ligand_waterfp.official_selection.run_official_selection`) into a small,
structured G1/G2 result - the format `ligand_waterfp.ligand_cv.build_ligand_cv`
consumes to construct the ligand-side PLUMED CV.

This step does not re-run or reinterpret the selection logic itself; it
only parses the two result lines
`ligand_waterfp.official_selection.run_official_selection` already produced
(passed via --out from that script, or piped directly) into structured
atom name/serial pairs:

    "anti-bulk fp selection: ATOM1 (5), ATOM2 (12)"  -> G1 = [(ATOM1, 5), (ATOM2, 12)]
    "bulk fp selection: ATOM3 (18), ATOM4 (19)"      -> G2 = [(ATOM3, 18), (ATOM4, 19)]

Usage:
    ligand-waterfp-g1g2 --selection-output run_official_selection.out \\
        --out outputs/selection/g1_g2.yaml

    # or pipe the two lines directly:
    ligand-waterfp-select ... | ligand-waterfp-g1g2 --stdin --out ...
"""
import argparse
import re
import sys
import os
import yaml

LINE_RE = re.compile(r"(anti-bulk|bulk) fp selection:\s*(\w+)\s*\((\d+)\),\s*(\w+)\s*\((\d+)\)")


def parse_selection_lines(text):
    """Returns {'G1': [{'name':..,'serial':..}, ...], 'G2': [...]}."""
    result = {}
    for m in LINE_RE.finditer(text):
        kind, name1, serial1, name2, serial2 = m.groups()
        key = "G1" if kind == "anti-bulk" else "G2"
        result[key] = [
            {"name": name1, "serial": int(serial1)},
            {"name": name2, "serial": int(serial2)},
        ]
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--selection-output", help="Path to the two result lines "
                      "written by run_official_selection.py's --out")
    src.add_argument("--stdin", action="store_true", help="Read the two result lines from stdin")
    p.add_argument("--system-id", default=None,
                    help="Optional free-text identifier stored alongside the result "
                         "(e.g. a run/config label) - not used in any calculation")
    p.add_argument("--out", required=True, help="Output YAML path, e.g. outputs/selection/g1_g2.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    text = sys.stdin.read() if args.stdin else open(args.selection_output).read()

    result = parse_selection_lines(text)
    if "G1" not in result or "G2" not in result:
        raise SystemExit(
            "Could not find both an 'anti-bulk fp selection: ...' and a "
            "'bulk fp selection: ...' line in the input. Got:\n" + text
        )

    payload = {"G1": result["G1"], "G2": result["G2"]}
    if args.system_id:
        payload["system_id"] = args.system_id

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"[select_g1_g2] G1 = {[a['name'] for a in result['G1']]}, "
          f"G2 = {[a['name'] for a in result['G2']]}")
    print(f"[select_g1_g2] wrote {args.out}")


if __name__ == "__main__":
    main()
