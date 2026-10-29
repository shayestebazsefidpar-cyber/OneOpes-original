"""The ``oneopes`` command: one subcommand per stage, each a typed function whose docstring is its help."""

from collections.abc import Sequence

from cyclopts import App

from oneopes.ligand.g1_g2 import select
from oneopes.ligand.monitor import monitor
from oneopes.ligand.plumed_cv import build_cv
from oneopes.ligand.setup import prepare_system
from oneopes.protein.analysis import protein
from oneopes.render.add_legend import add_legend

app = App(name="oneopes", help="Hydration and funnel collective variables for OneOPES, from unbiased MD.",
          result_action="return_value")
for command in (prepare_system, monitor, select, build_cv, protein, add_legend):
    app.command(command)


def main(argv: Sequence[str] | None = None) -> None:
    app(argv)
