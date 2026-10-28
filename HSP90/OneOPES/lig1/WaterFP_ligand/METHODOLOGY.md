# Methodology used in `WaterFP_ligand/` — HSP90-lig1 ligand-only hydration analysis

This folder contains the **ligand-side WaterFP workflow** used for HSP90-lig1.

The purpose of this workflow is to characterize the hydration behavior of the ligand atoms in a **ligand-only reference simulation**, identify the most water-perturbing and most bulk-like ligand regions, and use the resulting atom pairs as the **G1/G2 ligand-side inputs for reconstructing the corresponding PLUMED CV**.

---

# Workflow overview

```text
Ligand MOL
    ↓
Ligand-only solvated system
    ↓
Unbiased MD
    ↓
Water-oxygen RDF for every ligand heavy atom
    ↓
WaterFP for every ligand atom
    ↓
Block-wise convergence analysis
    ↓
Final converged FP ranking
    ↓
Official WaterFP atom-selection algorithm
    ↓
G1 / G2 ligand atom pairs
    ↓
Use G1/G2 to reconstruct the ligand-side PLUMED CV
```

---

# Stage 1 — Build and convergence-monitored ligand-only MD

## 1. System construction

The ligand was extracted from the original protein–ligand system as a standalone molecule:

```text
ligand_only.gro
```

The ligand was then:

1. solvated in TIP3P water;
2. neutralized / ionized as required;
3. energy minimized;
4. equilibrated using NVT;
5. equilibrated using NPT;
6. propagated using unbiased production MD.

The same general GROMACS conventions used elsewhere in the project were retained so that the ligand-only hydration reference remains consistent with the rest of the workflow.

The protein was **not present** in this reference simulation.

The ligand was identified as:

```text
resname MOL
```

---

# 2. Production convergence monitoring

Production MD was not treated as a blindly fixed-length simulation.

Instead, production was monitored online using:

```text
monitor_convergence.py
```

The production trajectory was divided into consecutive blocks and automatically stopped once the ligand hydration fingerprints satisfied the predefined convergence criteria.

The convergence test was performed for every ligand heavy atom.

For HSP90-lig1, the ligand contains 19 heavy atoms:

```text
N1
C1–C15
O1–O3
```

For consecutive production blocks, three criteria were evaluated.

### Criterion 1 — RDF shape stability

For every ligand atom, calculate the normalized RMSD between the water-oxygen RDF profiles of consecutive blocks.

Requirement:

```text
RDF nRMSD ≤ 15%
```

The exact implementation used by the convergence-monitoring code was retained.

---

### Criterion 2 — WaterFP stability

For every ligand atom, calculate the relative change in its WaterFP value between consecutive blocks.

Requirement:

```text
FP change ≤ 10%
```

---

### Criterion 3 — FP ranking stability

Rank all ligand heavy atoms according to their WaterFP values for each block.

Calculate the Spearman rank correlation between consecutive blocks.

Requirement:

```text
Spearman ρ ≥ 0.90
```

---

# 3. Consecutive-block convergence requirement

All three criteria must be satisfied simultaneously for:

```text
3 consecutive 5 ns block transitions
```

A single stable transition is therefore not sufficient to declare convergence.

The production simulation is stopped only after the required sequence of stable transitions has been observed.

---

# Stage 2 — WaterFP calculation

For every ligand heavy atom, calculate the water-oxygen radial density profile from the ligand-only trajectory.

The WaterFP value is calculated using the same WaterFP excess-entropy formulation used throughout this project:

```text
FP = ∫ -2π · norm · [g(r) ln(g(r)) - g(r) + 1] · r² dr
```

where:

- `g(r)` is the normalized water-oxygen radial density profile;
- `r` is the radial distance from the ligand atom;
- `norm` is the empirical bulk-density normalization.

For this workflow:

```text
norm = mean of the outermost 500 bins
```

of the 2001-bin RDF profile.

The same RDF range, binning, normalization convention, and FP definition used by the reference WaterFP implementation are preserved.

The FP calculation therefore represents a **hydration fingerprint** for each ligand atom.

---

# Stage 3 — Converged ligand FP ranking

Once convergence is reached, the final converged block is used to obtain the FP value for every ligand heavy atom.

The analysis produces:

- FP value for every ligand atom;
- FP ranking;
- atom identity;
- block-wise FP values;
- convergence metrics.

For HSP90-lig1, convergence was reached at:

```text
19.99 ns
```

This was substantially shorter than the originally planned 100 ns because the hydration fingerprints satisfied the predefined convergence criteria.

The observed convergence diagnostics were:

```text
worst FP change    = 2.3%
worst RDF nRMSD    = 8.5%
minimum Spearman ρ = 0.96
```

The FP ranking was already stable from the early production blocks.

These values describe the HSP90-lig1 analysis and are not assumed to be universal for other systems.

---

# Stage 4 — Official WaterFP atom-selection algorithm

After convergence, the **official WaterFP atom-selection procedure** was applied to the final converged FP ranking.

The selection code was taken from the published WaterFP implementation:

```text
github.com/valeriorizzi/WaterFP
```

The relevant selection functions were used directly:

```text
select_next_atom
select_bulk_atom
```

The published selection logic was retained rather than introducing a new project-specific atom-selection rule.

The workflow is therefore:

```text
MD + RDF
    ↓
WaterFP
    ↓
converged FP ranking
    ↓
official WaterFP selection algorithm
    ↓
G1 / G2
```

---

# Stage 5 — G1: anti-bulk ligand pair

The first ligand-side pair represents the region of the ligand with the strongest water perturbation.

The algorithm identifies the ligand atom with the strongest WaterFP value according to the final converged FP ranking.

For HSP90-lig1:

```text
G1 anchor = N1
```

The corresponding partner atom is then selected using the official WaterFP graph-distance / FP-grouping rule.

For HSP90-lig1:

```text
G1 partner = C8
```

Therefore:

```text
G1 = N1, C8
```

This pair represents the ligand region associated with the strongest hydration perturbation under the WaterFP atom-selection procedure.

---

# Stage 6 — G2: bulk-like ligand pair

The second ligand-side pair represents the most bulk-like / solvent-like region of the ligand.

The algorithm identifies the ligand atom with the weakest WaterFP value.

For HSP90-lig1:

```text
G2 anchor = O3
```

The corresponding partner is then selected using the official WaterFP bulk-selection rule.

For HSP90-lig1:

```text
G2 partner = O2
```

Therefore:

```text
G2 = O3, O2
```

---

# Final ligand-side result

For HSP90-lig1:

```text
G1 = N1, C8
G2 = O3, O2
```

These G1/G2 pairs are the **ligand-side WaterFP-derived atom groups** used as inputs for reconstructing the corresponding ligand-side CV in `plumed.dat`.

The resulting groups are generated from the converged ligand hydration fingerprints through the WaterFP atom-selection procedure.

---

# Reproducibility workflow

When applying this workflow to another ligand:

1. identify the ligand heavy atoms;
2. construct the ligand-only TIP3P reference system;
3. run unbiased MD;
4. monitor RDF, FP, and FP-ranking convergence;
5. stop only after the predefined consecutive-block convergence criterion is satisfied;
6. calculate the final converged WaterFP ranking;
7. apply the official WaterFP atom-selection implementation;
8. record the resulting G1 and G2 atom pairs;
9. use those atom groups to reconstruct the ligand-side PLUMED CV.

The same RDF methodology, FP definition, normalization convention, convergence criteria, and atom-selection procedure should be retained when applying the workflow to another system.
