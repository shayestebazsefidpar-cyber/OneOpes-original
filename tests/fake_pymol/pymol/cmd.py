"""PyMOL's cmd, standing in: every call is accepted and does nothing, except png, which writes a white image."""
from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.image import imsave


def png(filename: str, width: int, height: int, **_: Any) -> None:
    imsave(filename, np.ones((height, width, 3)))


def __getattr__(name: str) -> Callable[..., None]:
    return lambda *args, **kwargs: None
