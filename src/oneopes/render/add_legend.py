"""A legend on a raw render of the selected atoms (``oneopes add-legend``)."""
import logging
from pathlib import Path

from cyclopts.types import ExistingFile
from PIL import Image, ImageDraw, ImageFont

from oneopes.errors import InputError

logger = logging.getLogger(__name__)

DEFAULT_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def parse_entry(spec):
    """'label:rgb_csv' e.g. 'Anti-bulk (G1):ATOM1, ATOM2:220,20,20' -> (label, (r,g,b))."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise InputError(f"--entry must be 'title:detail:r,g,b', got: {spec}")
    title, detail, rgb_csv = parts
    r, g, b = (int(x) for x in rgb_csv.split(","))
    return f"{title}:  {detail}", (r, g, b)


def load_fonts(title_size, body_size):
    for path in DEFAULT_FONT_PATHS:
        try:
            return ImageFont.truetype(path, title_size), ImageFont.truetype(path, body_size)
        except Exception:
            continue
    return ImageFont.load_default(), ImageFont.load_default()


def add_legend(*, src: ExistingFile, dst: Path, entry: list[str], title: str = "Selected atoms", caption: str = "",
               title_font_size: int = 34, body_font_size: int = 30) -> None:
    """Add a legend to a raw render of the selected atoms and save the annotated figure.

    The raw render comes from render_selected_atoms.py (PyMOL) or render_selected_atoms_chimera.py.
    Example: --entry "Anti-bulk (G1):ATOM1, ATOM2:220,20,20" --entry "Bulk (G2):ATOM3, ATOM4:30,60,220".

    Parameters
    ----------
    src
        The raw render.
    dst
        The annotated figure to write.
    entry
        One legend swatch as 'label:detail:r,g,b'; repeat for each swatch.
    """
    entries = [parse_entry(e) for e in entry]

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
