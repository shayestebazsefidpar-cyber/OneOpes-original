"""The G1/G2 legend on a render of the selected atoms."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from oneopes.ligand.g1_g2 import G1G2Selection
from oneopes.render import G1_RGB, G2_RGB, RENDER_DPI, Rgb


def legend_entries(selection: G1G2Selection) -> list[tuple[str, Rgb]]:
    """The legend entries of a G1/G2 selection, coloured as the render colours them."""
    return [(f"Anti-bulk (G1):  {selection.g1_names}", G1_RGB), (f"Bulk (G2):  {selection.g2_names}", G2_RGB)]


def with_legend(image: np.ndarray, selection: G1G2Selection, caption: str = "", font_size: float = 8) -> Figure:
    """``image`` at its own size in pixels, with the G1/G2 legend in the top-right corner and ``caption`` below."""
    height, width = image.shape[:2]
    fig = Figure(figsize=(width / RENDER_DPI, height / RENDER_DPI), dpi=RENDER_DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.set_axis_off()
    handles = [Line2D([], [], marker="o", linestyle="", color=(r / 255, g / 255, b / 255), label=label)
               for label, (r, g, b) in legend_entries(selection)]
    ax.legend(handles=handles, title="Selected atoms", loc="upper right", fontsize=font_size,
              title_fontsize=font_size)
    if caption:
        fig.text(0.02, 0.02, caption, fontsize=font_size)
    return fig
