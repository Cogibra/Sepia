import unittest

from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from sepia.graph.functional import \
        GraphParametersWB, \
        graph_conv, \
        calc_graph_conv_auto_loss, \
        grad_graph_conv_auto_loss

from sepia.graph.data import parse_edges_to_adjacency

class TestGraphConv(unittest.TestCase):

    def setUp(self):

        pass

    def test_graph_conv(self):

        pass


if __name__ == "__main__":

    unittest.main(verbosity=1)
