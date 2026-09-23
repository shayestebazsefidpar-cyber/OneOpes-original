#!/usr/bin/bash

import numpy

fp=[]

for atom in ["C1", "C2", "C3", "C4", "C5", "C6", "O1", "O2"]:

	data = numpy.loadtxt(f"rdf_{atom}.xvg", comments=['#', '@'])

	x = (4/3)*numpy.pi*(data[:,0])**3 #x i 1e-3 nm
	y = data[:,1]

	for i, d in enumerate(y):
		if i == 0:
			pass
		else:
			x[i] = x[i] - x[i-1]

	norm = numpy.mean(y[-500:])
	g = y/norm

	mask = (g == 0)
	r = data[:,0]
	g[mask]  = -2*numpy.pi*norm*(r[mask])**2
	g[~mask] = -2*numpy.pi*norm*(g[~mask]*numpy.log(g[~mask])-g[~mask]+1)*r[~mask]**2

	fp.append(numpy.trapz(g, dx=1e-3))

with open("mol.pdb") as f:
    lines = f.readlines()

for i in range(4, 12):
    line = lines[i]
    start, end = 61, 67
    lines[i] = f"{line[:start]}{fp[i-4]:.2f}{line[end:]}"

with open("mol.pdb", "w") as f:
    f.writelines(lines)

