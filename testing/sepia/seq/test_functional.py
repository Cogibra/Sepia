import unittest

from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from sepia.seq.functional import \
        NICEParametersWB, \
        bijective_forward, \
        bijective_reverse


class TestNICE(unittest.TestCase):

    def setUp(self):

        weights = npr.randn(3, 4, 8)
        biases = jnp.zeros((3, 1, 12))

        self.parameters = NICEParametersWB(weights = weights, biases = biases)

    def test_bijective_forward(self):
        
        input_a = npr.rand(4, 12)
        input_b = npr.rand(4, 12)

        output_a = bijective_forward(input_a, self.parameters)
        output_b = bijective_forward(input_b, self.parameters)

        output_2a = bijective_forward(2.0 * input_a, self.parameters)

        output_ab = bijective_forward(input_a + input_b, self.parameters)
        
        super_2a = superposition_2a = 2.0 * output_a
        super_ab = superposition_ab = output_a + output_b

        self.assertNotAlmostEqual(jnp.abs(super_2a - output_2a).mean(), \
                0.0, places=4)

        self.assertNotAlmostEqual(jnp.abs(super_ab - output_ab).mean(), \
                0.0, places=4)

    def test_bijective_reverse(self):

        input_a = npr.rand(4, 12)
        input_b = npr.rand(4, 12)

        output_a = bijective_forward(input_a, self.parameters)
        output_b = bijective_forward(input_b, self.parameters)
        
        new_input_a = bijective_reverse(output_a, self.parameters)
        new_input_b = bijective_reverse(output_b, self.parameters)

        # precision error is typically on the order of 1e-8
        self.assertAlmostEqual(jnp.abs(new_input_a-input_a).mean(), \
                0.0, places=6)
        self.assertAlmostEqual(jnp.abs(new_input_b-input_b).mean(), \
                0.0, places=6)


if __name__ == "__main__":

    unittest.main(verbosity=1)
