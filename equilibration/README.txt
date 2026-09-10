Here are useful things to equilibrate the systems.

There are:
* The MDP files used (the same for every system), CG can help sometimes when dealing with off-center charges, but is mostly not needed
* An example SBATCH script, this script will also launch a longer unbiased run at the end from which to extract the sigma values for the CVs to insert in your plumed file (in the case of the plumed.dat files given here these values are already there and should not be modified)
* A directory called "position_restraints_for_equilibration" with the position restraints needed for the equilibration, always double check that the topology file has the right include statement

For any doubt about the equilibration process please check out this tutorial http://www.mdtutorials.com/gmx/complex/

