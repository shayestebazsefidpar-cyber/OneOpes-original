"""
protein_waterfp - protein-side WaterFP hydration analysis.

Modules:
    selection      - protein / ligand / water identification and the candidate pool
    trajectory     - block plan and the per-frame Contact / Bridge pass
    convergence    - block-wise WaterFP convergence (reuses ligand_waterfp)
    sasa           - static per-atom SASA (freesasa)
    final_table    - FP ranking and the final per-atom table
    cli            - ``protein-waterfp`` command
"""
__version__ = "0.1.0"
