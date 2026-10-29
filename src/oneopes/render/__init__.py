"""Figure of the selected atoms: a PyMOL render of the bound complex with a G1/G2 legend."""

Rgb = tuple[int, int, int]
G1_RGB: Rgb = (220, 20, 20)
"""G1 (anti-bulk) is red, in the render and its legend."""
G2_RGB: Rgb = (30, 60, 220)
"""G2 (bulk) is blue, in the render and its legend."""
RENDER_SIZE = (2400, 1800)
"""Width and height (px) of the render."""
RENDER_DPI = 300
