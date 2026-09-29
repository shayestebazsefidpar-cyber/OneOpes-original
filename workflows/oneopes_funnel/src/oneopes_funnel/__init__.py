"""
oneopes_funnel - funnel construction for OneOPES.

Modules:
    system_io     - load the protein-ligand system; identify protein / ligand / water
    p0_p1         - P0/P1 (atoms0/atoms1) selection and their COMs
    geometry      - funnel axis, perpendicular basis, extent profile, wall points
    plumed_io     - read-only parsing of existing plumed.dat funnel definitions
    make_funnel   - top-level makeFunnel-type API

Implemented (wrapping BioSimSpace): p0_p1, geometry, make_funnel. Stubs: system_io, plumed_io.
"""
__version__ = "0.1.0"
