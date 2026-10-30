"""The one exception type for problems with what the user gave: options, input files, the system in them."""


class InputError(ValueError):
    """Invalid options or inputs. ``oneopes`` reports it as a one-line error, without a traceback."""
