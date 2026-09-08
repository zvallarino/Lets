"""A stronger neighboring-fiber regression: the far edge is inside the ray."""
import unittest
import numpy as np
from scipy import ndimage as ndi
from fiber_diameter import measure_at


class NeighborEdgeRegression(unittest.TestCase):
    def test_stronger_neighbor_far_edge_is_not_selected(self):
        a=np.full((160,160),.05)
        a[65:85]=.5
        a[100:110]=.95  # Both this neighbor's edges lie inside the 40-px ray.
        m=measure_at(ndi.gaussian_filter(a,1),a>.2,75,80,np.array([0.,1.]),10)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m['diameter_px'],20,delta=1)


if __name__=='__main__':unittest.main(verbosity=2)
