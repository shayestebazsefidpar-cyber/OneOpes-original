"""``oneopes``: one command with a subcommand per stage; ``oneopes <subcommand> --help`` describes each."""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from collections.abc import Sequence

from oneopes.ligand import g1_g2, monitor, plumed_cv, setup
from oneopes.protein import analysis
from oneopes.render import add_legend

SUBCOMMANDS = {
    "prepare-system": setup,
    "monitor": monitor,
    "select": g1_g2,
    "build-cv": plumed_cv,
    "protein": analysis,
    "add-legend": add_legend,
}
"""Each module provides ``add_arguments(parser)`` and ``run(args)``; its docstring is the subcommand's help."""


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(prog="oneopes", description=__doc__)
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    for name, module in SUBCOMMANDS.items():
        summary = module.__doc__.strip().splitlines()[0]
        p = subcommands.add_parser(name, help=summary, description=module.__doc__,
                                   formatter_class=RawDescriptionHelpFormatter)
        module.add_arguments(p)
        p.set_defaults(run=module.run)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.run(args)
