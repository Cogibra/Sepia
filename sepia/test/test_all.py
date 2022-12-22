import unittest

from sepia.test.test_common import TestCommon
from sepia.test.graph.test_functional import TestGraphConv
from sepia.test.seq.test_functional import TestNICE,\
        TestSelfAttention,\
        TestEncoder,\
        TestDecoder

if __name__ == "__main__":
    
    unittest.main(verbosity=2)
