"""Command-line arguments shared by several ``oneopes`` subcommands."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path

from oneopes.convergence import DEFAULT_TOLERANCES, Tolerances
from oneopes.waterfp import NORM_TAIL_BINS_DEFAULT, RadialGrid

BLOCK_NS_DEFAULT = 5.0


def existing_file(path: str) -> Path:
    """Argument type for an input file: the path, or a parse error naming it when it does not exist."""
    file = Path(path)
    if not file.is_file():
        raise ArgumentTypeError(f"file not found: {file.resolve()}")
    return file


def add_system_arguments(p: ArgumentParser) -> None:
    """--ligand-selection and --water-selection (see :func:`oneopes.system.select_system`)."""
    p.add_argument("--ligand-selection", default=None, help="MDAnalysis selection (default: auto-detect)")
    p.add_argument("--water-selection", default=None, help="MDAnalysis selection (default: known water names)")


def add_grid_arguments(p: ArgumentParser) -> None:
    """The radial grid and bulk normalisation of WaterFP."""
    p.add_argument("--rmax-nm", type=float, default=RadialGrid.rmax_nm)
    p.add_argument("--binwidth-nm", type=float, default=RadialGrid.binwidth_nm)
    p.add_argument("--norm-tail-bins", type=int, default=NORM_TAIL_BINS_DEFAULT)


def radial_grid(args: Namespace) -> RadialGrid:
    return RadialGrid(rmax_nm=args.rmax_nm, binwidth_nm=args.binwidth_nm)


def add_convergence_arguments(p: ArgumentParser) -> None:
    """Block size and the stability tolerances between consecutive blocks."""
    tol = DEFAULT_TOLERANCES
    p.add_argument("--block-ns", type=float, default=BLOCK_NS_DEFAULT,
                   help=f"trajectory block size in ns (default: {BLOCK_NS_DEFAULT})")
    p.add_argument("--fp-rel-tol", type=float, default=tol.fp_rel_tol,
                   help=f"max relative FP change between blocks (default: {tol.fp_rel_tol})")
    p.add_argument("--rdf-nrmsd-tol", type=float, default=tol.rdf_nrmsd_tol,
                   help=f"max normalised RMSD between consecutive g(r) (default: {tol.rdf_nrmsd_tol})")
    p.add_argument("--spearman-tol", type=float, default=tol.spearman_tol,
                   help=f"min Spearman correlation between consecutive FP rankings (default: {tol.spearman_tol})")
    p.add_argument("--n-stable", type=int, default=tol.n_stable,
                   help=f"stable transitions in a row that mean converged (default: {tol.n_stable})")


def tolerances(args: Namespace) -> Tolerances:
    return Tolerances(fp_rel_tol=args.fp_rel_tol, rdf_nrmsd_tol=args.rdf_nrmsd_tol,
                      spearman_tol=args.spearman_tol, n_stable=args.n_stable)
