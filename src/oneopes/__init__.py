"""Hydration and funnel collective variables for OneOPES, from unbiased MD.

pymbar logs notices when it is imported (JAX is missing; read statistical inefficiencies with care);
they are silenced here, before any module imports it.
"""
import logging

logging.getLogger("pymbar").setLevel(logging.ERROR)

__version__ = "0.1.0"
