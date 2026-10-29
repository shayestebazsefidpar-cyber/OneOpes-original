"""The ``oneopes`` command: one subcommand per stage, each a typed function whose docstring is its help."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from cyclopts.config import Yaml

from oneopes.errors import InputError
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


@app.meta.default
def with_config(*tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
                config: Path | None = None):
    """Run a subcommand, taking its settings from ``config`` where the command line does not give them.

    Parameters
    ----------
    config
        YAML file with one section per subcommand, e.g. ``monitor: {block-ns: 2.0, n-stable: 4}``
        (see configs/example.yaml).
    """
    app.config = () if config is None else Yaml(config, must_exist=True)
    return app(tokens)


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``oneopes``, logging its progress (and other libraries' warnings) to stderr.

    An InputError ends the run with its message and exit code 1; any other exception keeps its traceback.
    """
    logging.basicConfig(format="[%(module)s] %(message)s", level=logging.WARNING)
    logger = logging.getLogger("oneopes")
    logger.setLevel(logging.INFO)
    try:
        app.meta(argv)
    except InputError as e:
        logger.error("error: %s", e)
        raise SystemExit(1) from None
