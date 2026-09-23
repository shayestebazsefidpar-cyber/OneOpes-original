"""
ligand_waterfp - ligand-only WaterFP hydration-CV workflow.

Subpackages, in pipeline order:
    system_setup        - build a standalone ligand-in-water MD system
    convergence          - block-wise hydration convergence monitor
    waterfp              - RDF + WaterFP fingerprint calculation
    official_selection   - vendored upstream WaterFP atom-selection algorithm
    g1_g2_selection       - formats the raw selection result into structured G1/G2
    ligand_cv             - builds a starting-point PLUMED CV fragment from G1/G2
    visualization         - optional: render selected atoms on the bound complex

See the top-level README.md for the full pipeline and usage.
"""
__version__ = "0.1.0"
