"""
ligand_waterfp - ligand-only WaterFP hydration-CV workflow.

Subpackages, in pipeline order:
    system_setup        - build a standalone ligand-in-water MD system
    convergence          - block-wise hydration convergence monitor
    g1_g2_selection       - WaterFP over the converged block, then G1/G2 atom selection
    official_selection   - vendored upstream WaterFP selection code, the tests' reference
    ligand_cv            - builds a starting-point PLUMED CV fragment from G1/G2
    visualization         - optional: render selected atoms on the bound complex

See the top-level README.md for the full pipeline and usage.
"""
__version__ = "0.1.0"
