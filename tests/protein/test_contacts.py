"""Contacts on hand-placed atoms: two candidates, one ligand atom, one water oxygen, two frames."""
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from oneopes.protein.contacts import Contacts


def test_contact_and_bridge_occupancy():
    """Candidate 0 touches the ligand in both frames and is bridged in the first only; candidate 1 is far away."""
    u = mda.Universe.empty(4, trajectory=True)
    near, far, ligand, water = (0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (4.0, 0.0, 0.0), (2.0, 0.0, 0.0)
    frames = [[near, far, ligand, water], [near, far, ligand, (2.0, 10.0, 0.0)]]
    u.load_new(np.array(frames, dtype=np.float32), format=MemoryReader, dimensions=[50, 50, 50, 90, 90, 90])

    results = Contacts(u.atoms[[0, 1]], u.atoms[[2]], u.atoms[[3]]).run().results

    np.testing.assert_array_equal(results.contact_occupancy, [1.0, 0.0])
    np.testing.assert_array_equal(results.bridge_occupancy, [0.5, 0.0])
