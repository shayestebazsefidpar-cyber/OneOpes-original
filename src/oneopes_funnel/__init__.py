"""
oneopes_funnel - funnel construction for OneOPES.

Modules:
    system_io     - load the protein-ligand system; identify protein / ligand / water
    p0_p1         - BioSimSpace makeFunnel() atom selections + derived P0/P1
    geometry      - Funnel CV parameters, getExtent() radius, wall points
    plumed_io     - read-only parsing of existing plumed.dat funnel definitions
    make_funnel   - top-level makeFunnel-type API

Implemented on top of the public BioSimSpace API:
    p0_p1, geometry, make_funnel
Stubs: system_io, plumed_io.
"""
__version__ = "0.1.0"
