"""
Composite a legend onto a raw render (from render_selected_atoms.py or
render_selected_atoms_chimera.py) and save the final annotated figure.

Legend entries and caption text are passed as arguments - nothing about a
specific system, ligand, or set of selected atoms is hardcoded here.

Usage:
    ligand-waterfp-add-legend --src selected_atoms_raw.png --dst selected_atoms.png \\
        --title "WaterFP selected atoms" \\
        --entry "Anti-bulk (G1):ATOM1, ATOM2:220,20,20" \\
        --entry "Bulk (G2):ATOM3, ATOM4:30,60,220" \\
        --caption "grey = protein pocket, yellow = ligand"
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def parse_entry(spec):
    """'label:rgb_csv' e.g. 'Anti-bulk (G1):ATOM1, ATOM2:220,20,20' -> (label, (r,g,b))."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--entry must be 'title:detail:r,g,b', got: {spec}")
    title, detail, rgb_csv = parts
    r, g, b = (int(x) for x in rgb_csv.split(","))
    return f"{title}:  {detail}", (r, g, b)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--title", default="Selected atoms")
    p.add_argument("--entry", action="append", required=True,
                    help="'label:detail:r,g,b', repeatable - one per legend swatch")
    p.add_argument("--caption", default="")
    p.add_argument("--title-font-size", type=int, default=34)
    p.add_argument("--body-font-size", type=int, default=30)
    return p.parse_args()


def load_fonts(title_size, body_size):
    for path in DEFAULT_FONT_PATHS:
        try:
            return ImageFont.truetype(path, title_size), ImageFont.truetype(path, body_size)
        except Exception:
            continue
    return ImageFont.load_default(), ImageFont.load_default()


def main():
    args = parse_args()
    entries = [parse_entry(e) for e in args.entry]

    img = Image.open(args.src).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    font_title, font_body = load_fonts(args.title_font_size, args.body_font_size)

    pad = 24
    sw = 28  # swatch size
    box_w = 560
    box_h = pad * 2 + 40 + len(entries) * 50 + 10
    x0, y0 = W - box_w - 40, 40
    x1, y1 = x0 + box_w, y0 + box_h

    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    draw.text((x0 + pad, y0 + pad), args.title, font=font_title, fill=(0, 0, 0))

    ty = y0 + pad + 50
    for label, rgb in entries:
        draw.ellipse([x0 + pad, ty, x0 + pad + sw, ty + sw], fill=rgb)
        draw.text((x0 + pad + sw + 14, ty - 2), label, font=font_body, fill=(0, 0, 0))
        ty += 50

    if args.caption:
        draw.text((40, H - 60), args.caption, font=font_body, fill=(60, 60, 60))

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    img.save(args.dst, dpi=(300, 300))
    print(f"[add_legend] wrote {args.dst}")


if __name__ == "__main__":
    main()
