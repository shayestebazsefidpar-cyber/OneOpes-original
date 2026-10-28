"""
protein_waterfp - protein-side WaterFP hydration analysis.

Modules:
    selection      - protein / ligand / water identification and the candidate pool
    trajectory     - the per-frame Contact / Bridge pass
    sasa          - static per-atom SASA (freesasa)
    final_table    - the final per-atom table
    cli            - ``protein-waterfp`` command
"""
__version__ = "0.1.0"
