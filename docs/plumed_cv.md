# Stage 4 - PLUMED fragment

`oneopes build-cv` writes a PLUMED fragment from `g1_g2.yaml`: a `GROUP` for G1 and G2 and a
`COORDINATION` of each with a water-oxygen group.

```bash
oneopes build-cv --g1-g2 g1_g2.yaml --out plumed_fragment.dat [--water-group-def "GROUPA=1305-14858"]
```

It is a starting point, not a ready input. Before use:

1. Define the water-oxygen group (`--water-group-def`) or reuse one from your `plumed.dat`.
2. Set `R_0`, `NN`, `MM` (placeholders: 0.32, 6, 12) to match your other CVs.
3. Merge it into your `plumed.dat` by hand; nothing existing is overwritten.
