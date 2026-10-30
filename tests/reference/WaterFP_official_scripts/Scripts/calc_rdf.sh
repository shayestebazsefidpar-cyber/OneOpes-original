#! /usr/bin/bash

gmx make_ndx -f nvt.gro <<EOF
2 & ! a H*
name 7 m
5 & a OW
name 8 OW
splitat 7
q
EOF

gmx trjconv -f ../example_S6-G5/50000.xtc -pbc mol -center -o pbc.xtc -s topol.tpr<<EOF
2
0
EOF

declare -a arr=("C1" "C2" "C3" "C4" "C5" "C6" "O1" "O2")

for i in "${!arr[@]}"; do
		gmx rdf -f pbc.xtc -n index.ndx -ref m_"${arr[i]}"_"$((i+1))" -sel OW -o rdf_"${arr[i]}".xvg -bin 0.001 -norm number_density -rmax 2.001
		echo calculating the water oxygen rdf of "${arr[i]}"
		echo m_"${arr[i]}"_"${i}"
done

rm pbc.xtc

gmx editconf -f nvt.gro -n index.ndx -o mol.pdb << EOF
2
EOF

rm index.ndx

python3 fp.py
