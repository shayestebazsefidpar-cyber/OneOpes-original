"""
Composite a legend onto the raw Chimera render, and save the final
publication figure at the requested path/filename.
"""
from PIL import Image, ImageDraw, ImageFont
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "HSP90_lig1_WaterFP_selected_atoms_raw.png")
DST = os.path.join(HERE, "HSP90_lig1_WaterFP_selected_atoms.png")

img = Image.open(SRC).convert("RGB")
draw = ImageDraw.Draw(img)

W, H = img.size

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
except Exception:
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

pad = 24
box_w, box_h = 560, 190
x0, y0 = W - box_w - 40, 40
x1, y1 = x0 + box_w, y0 + box_h

draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(60, 60, 60), width=2)

draw.text((x0 + pad, y0 + pad), "WaterFP selected atoms", font=font_title, fill=(0, 0, 0))

sw = 28  # swatch size
ty = y0 + pad + 50
draw.ellipse([x0 + pad, ty, x0 + pad + sw, ty + sw], fill=(220, 20, 20))
draw.text((x0 + pad + sw + 14, ty - 2), "Anti-bulk:  N1, C8", font=font_body, fill=(0, 0, 0))

ty2 = ty + 50
draw.ellipse([x0 + pad, ty2, x0 + pad + sw, ty2 + sw], fill=(30, 60, 220))
draw.text((x0 + pad + sw + 14, ty2 - 2), "Bulk:  O3, O2", font=font_body, fill=(0, 0, 0))

# small caption bottom-left
cap = "HSP90 - lig1 bound complex; grey = protein pocket, yellow = ligand"
draw.text((40, H - 60), cap, font=font_body, fill=(60, 60, 60))

img.save(DST, dpi=(300, 300))
print("Saved final figure to: " + DST)
