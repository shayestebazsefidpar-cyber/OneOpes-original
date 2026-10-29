"""A legend on a raw render of the selected atoms (``oneopes add-legend``)."""
import logging
from pathlib import Path

from cyclopts.types import ExistingFile
from PIL import Image, ImageDraw, ImageFont

from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import SelectionResult, read_g1_g2_yaml

logger = logging.getLogger(__name__)

DEFAULT_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

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


def selection_entries(selection: SelectionResult) -> list[tuple[str, Rgb]]:
    """The legend entries of a G1/G2 selection, coloured as the renders colour them."""
    return [(f"Anti-bulk (G1):  {selection.G1[0].name}, {selection.G1[1].name}", G1_RGB),
            (f"Bulk (G2):  {selection.G2[0].name}, {selection.G2[1].name}", G2_RGB)]


def load_fonts(title_size: int, body_size: int) -> tuple[ImageFont.ImageFont | ImageFont.FreeTypeFont, ...]:
    """DejaVu Sans at the two sizes, or PIL's default font when it is not installed."""
    for path in DEFAULT_FONT_PATHS:
        try:
            return ImageFont.truetype(path, title_size), ImageFont.truetype(path, body_size)
        except OSError:
            continue
    return ImageFont.load_default(), ImageFont.load_default()


def add_legend(*, src: ExistingFile, dst: Path, g1_g2: ExistingFile | None = None, entry: list[str] | None = None,
               title: str = "Selected atoms", caption: str = "", title_font_size: int = 34,
               body_font_size: int = 30) -> None:
    """Add a legend to a raw render of the selected atoms and save the annotated figure.

    The raw render comes from render_selected_atoms.py (PyMOL) or render_selected_atoms_chimera.py.
    Example: --g1-g2 g1_g2.yaml, or --entry "Anti-bulk (G1):ATOM1, ATOM2:220,20,20" --entry "...".

    Parameters
    ----------
    src
        The raw render.
    dst
        The annotated figure to write.
    g1_g2
        g1_g2.yaml whose G1 (red) and G2 (blue) atoms become the first legend entries.
    entry
        One more legend swatch as 'label:detail:r,g,b'; repeat for each swatch.
    """
    entries = selection_entries(read_g1_g2_yaml(g1_g2)) if g1_g2 else []
    entries += [parse_entry(spec) for spec in entry or []]
    if not entries:
        raise InputError("The legend needs --g1-g2 or at least one --entry.")

    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    font_title, font_body = load_fonts(title_font_size, body_font_size)

    pad = 24
    swatch = 28
    box_w = 560
    box_h = pad * 2 + 40 + len(entries) * 50 + 10
    x0, y0 = width - box_w - 40, 40
    x1, y1 = x0 + box_w, y0 + box_h

    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    draw.text((x0 + pad, y0 + pad), title, font=font_title, fill=(0, 0, 0))

    ty = y0 + pad + 50
    for label, rgb in entries:
        draw.ellipse([x0 + pad, ty, x0 + pad + swatch, ty + swatch], fill=rgb)
        draw.text((x0 + pad + swatch + 14, ty - 2), label, font=font_body, fill=(0, 0, 0))
        ty += 50

    if caption:
        draw.text((40, height - 60), caption, font=font_body, fill=(60, 60, 60))

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, dpi=(300, 300))
    logger.info("wrote %s", dst)
