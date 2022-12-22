import unittest

import jax
from jax import numpy as jnp
from jax import grad

import numpy as np
import numpy.random as npr

from sepia.common import query_kwargs

class TestCommon(unittest.TestCase):
    
    def setUp(setUp):
        pass

    def test_query_kwargs(self):

        kwargs = {}

        keys = ["my_string",\
                "my_int",\
                "my_float",\
                "my_array"]

        defaults = ["default_string",\
                13,\
                42.0,\
                npr.rand(32,1)]

        my_args = ["arg_string",\
                1337,\
                43.0,\
                npr.rand(32,1)]

        for key, arg, default in zip(keys, my_args, defaults):

            not_arg = query_kwargs(key, default, **kwargs)
            kwargs[key] = arg
            not_default = query_kwargs(key, default, **kwargs)

            if type(default) == np.ndarray:
                self.assertEqual(0.0, np.sum(arg - not_default))
                self.assertEqual(0.0, np.sum(default - not_arg))

                self.assertNotEqual(0.0, np.sum(arg - not_arg))
                self.assertNotEqual(0.0, np.sum(default - not_default))
            else:
                self.assertNotEqual(arg, not_arg)
                self.assertNotEqual(default, not_default)

                self.assertEqual(arg, not_default)
                self.assertEqual(default, not_arg)
            
if __name__ == "__main__":

    unittest.main(verbosity=1)
