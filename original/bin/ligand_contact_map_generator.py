#!/bin/python

'''
(c) 2019 Antonija Kuzmanic

'''

# Usage:
# python ligand_contact_map_generator.py -f ../ligand2_whole.pdb -lig-selection-string "resname LIG and not element H" -cutoff 0.45

import argparse as ap
import textwrap

import mdtraj

apolar_residues = ["GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "MET", "TRP"]

parser = ap.ArgumentParser(
    add_help=False,
    formatter_class=ap.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        The script creates a PLUMED contact map
        between the protein and the ligand

        The script requires the following inputs:
        - pdb or gro file with the reference structure

        The output name is optional (default - cmap.dat).
        '''),
    epilog=textwrap.dedent('''\
        WARNING:

        The script will not work for very large systems where
        atom number counter resets to 1 after reaching the space
        limit of the gro file.
        '''))

# Define required arguments.
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-f', help=('reference structure file - ' +
                                       '.gro or .pdb format'),
                           dest='fstruct', required=True, type=str)

required_args.add_argument('-lig-selection-string', help='The mdtraj selection string for the ligand without hydrogens'
                            'for example resname LIG and not element H',
                            dest='lig_selection_string', type=str, required=True)

# Define optional arguments.
optional_args = parser.add_argument_group('optional arguments')
optional_args.add_argument('-h', '--help', action='help',
                           help='show this help message and exit')
optional_args.add_argument('-o', help='output PLUMED cmap file', dest='fout',
                           default='cmap.dat',
                           type=str)
optional_args.add_argument('-vis', dest='fpml', default='show_dists.pml',
                           help='output PyMOL script to visualise distances',
                           type=str)
optional_args.add_argument('-cutoff', help='contact distance cutoff (nm)',
                           dest='cutoff', default=0.3, type=float)

optional_args.add_argument('-include', help='protein residue types to include in the '
                                            'contact map, the possible options are all (default), apolar or a comma separeted list of resnames '
                                            f'the using apolar is like using {",".join(apolar_residues)}',
                           dest='include', type=str, default="all")

optional_args.add_argument('-max-contacts', help='Maximum number of contacts to use (default=5)',
                           dest='max_contacts', default=5, type=int)


args = parser.parse_args()  # parse arguments


def switch(r, r0, d0=0, n=4, m=10):
    """
    Calculate the rational switching function where d0 is the point around
    which the function is 1, while r0 represents an inflexion point of the
    function and for values r>r0>d0 and r<r0<d0, the function quickly decays to
    zero.
    https://www.plumed.org/doc-v2.5/user-doc/html/switchingfunction.html
    """
    if r0 == (r-d0):
        # to avoid divisions with 0
        r = r + 0.000000001
    frac = (r-d0)/r0
    s = (1-frac**n)/(1-frac**m)
    return s


def create_cmap(atom_1_2_dist, resSeq_to_show, max_contacts):
    """
    Create a contact map following Federico's approach in his JCTC paper
    (DOI: 10.1021/acs.jctc.8b00263). He used d0=0, n=6, m=12, and r0=0.35 for
    backbone contacts and r0=0.55 for the rest. Also output a PyMOL script for
    the visualisation of distances.
    """

    with open(args.fout, "w") as fout:
        with open(args.fpml, "w") as fpml:

            fout.write('CONTACTMAP ...\n')
            fpml.write('set cartoon_side_chain_helper, off\n')
            fpml.write('load {}\n'.format(args.fstruct))

            # Loop through the pairwise distances data frame.
            j=0
            for atom_1, atom_2, _ in atom_1_2_dist:

                j +=1

                # Set the r0 according to the atom type.
                #if (d['atm_name1'] in hb_bckbn or d['atm_name2'] in hb_bckbn):
                #    r0 = 0.35
                #else:
                #    r0 = 0.55
                r0 = 0.55
                # Calculate and write out the contact map.
                fout.write(
                    '{:18}      {:>9}{}  \n'
                    .format('ATOMS{}={},{}'.format(j, atom_1, atom_2),
                            'SWITCH{}'.format(j),
                            '={{RATIONAL R_0={} D_0=0.00 NN=4 MM=10}}'.format(r0))
                )
                # Write out the distance part of the PyMOL visualisation script.
                fpml.write('distance id {}, id {}\n'.format(atom_1, atom_2))

                if j >= max_contacts:
                    break
                
            print('Creating the contact map with {} distances.'.format(j))
            if j > 200:
                print('WARNING! The number of distances is larger than 200. ' +
                    'Consider reducing their number to speed up the simulations.')
            # Finishing bits of both outfiles.
            fout.write('LABEL=cmap\nSUM\n... CONTACTMAP')
            fpml.write(f'show sticks, resi {"+".join(resSeq_to_show)} and not name h*\n')
            
            fout.write('\n\nCONTACTMAP ...\n')
            j=0
            for atom_1, atom_2, _ in atom_1_2_dist:

                j +=1

                # Set the r0 according to the atom type.
                #if (d['atm_name1'] in hb_bckbn or d['atm_name2'] in hb_bckbn):
                #    r0 = 0.35
                #else:
                #    r0 = 0.55
                r0 = 0.55
                # Calculate and write out the contact map.
                fout.write(
                    '{:18}      {:>9}{}  \n'
                    .format('ATOMS{}={},{}'.format(j, atom_1, atom_2),
                            'SWITCH{}'.format(j),
                            '={{RATIONAL R_0={} D_0=0.00 NN=4 MM=10}}'.format(r0))
                )
                # Write out the distance part of the PyMOL visualisation script.
                fpml.write('distance id {}, id {}\n'.format(atom_1, atom_2))

                if j >= max_contacts:
                    break
                

            # Finishing bits of both outfiles.
            fout.write('LABEL=cmappa\n... CONTACTMAP')

            print('Done!')


if __name__ == '__main__':

    pdb_file = args.fstruct

    if args.include == "all":
        apolar_residues = False
    elif args.include == "apolar":
        # do nothing
        apolar_residues = apolar_residues
    else:
        apolar_residues = args.include.split(",")

    traj = mdtraj.load(pdb_file)

    lig_atoms = mdtraj.load(pdb_file).top.select(args.lig_selection_string)

    # Keep only the interesting heavy atoms
    if apolar_residues:
        protein_atoms = mdtraj.load(pdb_file).top.select("(resname " + "or resname ".join(apolar_residues) + ") and (not element H)")
    else:
        protein_atoms = mdtraj.load(pdb_file).top.select("protein and not element H")


    neighbors_atoms = mdtraj.geometry.neighbors.compute_neighbors(traj,
                                                                cutoff=args.cutoff,
                                                                query_indices=traj.top.select(args.lig_selection_string),
                                                                haystack_indices=protein_atoms)[0]

    # I only want one atom per neighboring residue
    neighbors_resSeq = set([traj.top.atom(atom).residue.resSeq for atom in neighbors_atoms])

    neighbors_atoms = []
    for resSeq in neighbors_resSeq:
        atom_1, atom_2, dist = mdtraj.geometry.distance.find_closest_contact(traj,
                                                                        group1=traj.top.select(f"resSeq {resSeq} and not element H"),
                                                                        group2=traj.top.select(args.lig_selection_string))
        
        neighbors_atoms.append((traj.top.atom(atom_1).serial, traj.top.atom(atom_2).serial, dist))

    # Sort according to dist
    neighbors_atoms.sort(key=lambda x:x[-1])

    traj = traj.atom_slice(traj.top.select(f"protein or ({args.lig_selection_string})"))

    resSeq_to_show = list(set([residue.resSeq for residue in traj.top.residues]))

    resSeq_to_show.sort()

    resSeq_to_show = [str(i) for i in resSeq_to_show]

    create_cmap(neighbors_atoms, resSeq_to_show, args.max_contacts)
