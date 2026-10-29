"""
Funnel construction for OneOPES, on top of the public BioSimSpace API.

Modules:
    bss           - the BioSimSpace calls, with lengths and positions as plain numbers
    p0_p1         - BioSimSpace makeFunnel() atom selections + derived P0/P1
    geometry      - Funnel CV parameters, getExtent() radius, wall points
    make_funnel   - selections + the BioSimSpace Funnel CV
"""
