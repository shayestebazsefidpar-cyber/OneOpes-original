"""A legend on a raw render of the selected atoms (``oneopes add-legend``)."""
import logging
from pathlib import Path

from cyclopts.types import ExistingFile
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.lines import Line2D

from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import G1G2Selection, read_g1_g2_yaml

logger = logging.getLogger(__name__)

Rgb = tuple[int, int, int]
G1_RGB: Rgb = (220, 20, 20)
"""G1 (anti-bulk) is red in the renders."""
G2_RGB: Rgb = (30, 60, 220)
"""G2 (bulk) is blue in the renders."""


def parse_entry(spec: str) -> tuple[str, Rgb]:
    """'title:detail:r,g,b', e.g. 'Anti-bulk (G1):ATOM1, ATOM2:220,20,20' -> (label, (r, g, b))."""
    try:
        title, detail, rgb_csv = spec.split(":")
        r, g, b = (int(x) for x in rgb_csv.split(","))
    except ValueError:
        raise InputError(f"--entry must be 'title:detail:r,g,b', got: {spec}") from None
    return f"{title}:  {detail}", (r, g, b)


def selection_entries(selection: G1G2Selection) -> list[tuple[str, Rgb]]:
    """The legend entries of a G1/G2 selection, coloured as the renders colour them."""
    return [(f"Anti-bulk (G1):  {selection.g1_names}", G1_RGB), (f"Bulk (G2):  {selection.g2_names}", G2_RGB)]


def add_legend(*, src: ExistingFile, dst: Path, g1_g2: ExistingFile | None = None, entry: list[str] | None = None,
               title: str = "Selected atoms", caption: str = "", font_size: float = 8, dpi: int = 300) -> None:
    """Add a legend to a raw render of the selected atoms and save the annotated figure.

    The raw render comes from render_selected_atoms.py (PyMOL) or render_selected_atoms_chimera.py.
    Example: --g1-g2 g1_g2.yaml, or --entry "Anti-bulk (G1):ATOM1, ATOM2:220,20,20" --entry "...".

    Parameters
    ----------
    src
        The raw render.
    dst
        The annotated figure to write, with the size in pixels of the render.
    g1_g2
        g1_g2.yaml whose G1 (red) and G2 (blue) atoms become the first legend entries.
    entry
        One more legend entry as 'label:detail:r,g,b'; repeat for each entry.
    title
        Title of the legend.
    caption
        Text along the bottom of the figure (default: none).
    font_size
        Font size (pt) of the legend and the caption.
    dpi
        Resolution of the figure, which sets how large a point is in pixels.
    """
    entries = selection_entries(read_g1_g2_yaml(g1_g2)) if g1_g2 else []
    entries += [parse_entry(spec) for spec in entry or []]
    if not entries:
        raise InputError("The legend needs --g1-g2 or at least one --entry.")

    image = imread(src)
    height, width = image.shape[:2]
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.set_axis_off()
    handles = [Line2D([], [], marker="o", linestyle="", color=(r / 255, g / 255, b / 255), label=label)
               for label, (r, g, b) in entries]
    ax.legend(handles=handles, title=title, loc="upper right", fontsize=font_size, title_fontsize=font_size)
    if caption:
        fig.text(0.02, 0.02, caption, fontsize=font_size)

    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=dpi)
    logger.info("wrote %s", dst)
