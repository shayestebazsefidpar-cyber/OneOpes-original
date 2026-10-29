"""The BioSimSpace the funnel package uses, with its quantities as plain numbers (lengths in nm, positions in A)."""

import numpy as np


def require_biosimspace():
    """Import BioSimSpace or fail with an actionable message."""
    try:
        import BioSimSpace  # pyright: ignore[reportMissingImports]
    except ImportError as err:
        raise ImportError(
            "oneopes.funnel needs BioSimSpace (conda-only, e.g. the 'biosimspace' "
            "conda env); it is not pip-installable."
        ) from err
    return BioSimSpace


def nanometers(length) -> float:
    """A BioSimSpace ``Length`` in nm."""
    return length.nanometers().value()


def length_nm(value_nm: float):
    """A BioSimSpace ``Length`` of ``value_nm`` nm."""
    require_biosimspace()
    from BioSimSpace.Types import Length  # pyright: ignore[reportMissingImports]

    return Length(float(value_nm), "nanometer")


def atom_position_A(system, index: int, property_map: dict | None = None) -> np.ndarray:
    """Position (A) of system atom ``index``, via the public ``Atom.coordinates()``."""
    position = system.getAtom(int(index)).coordinates(property_map=property_map or {})
    return np.array([position.x().angstroms().value(), position.y().angstroms().value(),
                     position.z().angstroms().value()])
