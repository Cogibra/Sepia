import unittest

import sepia.seq.transformer

from sepia.test.test_common import TestCommon
from sepia.test.graph.test_functional import TestGraphConv
from sepia.test.seq.test_functional import TestNICE,\
        TestMultiHead,\
        TestSelfAttention,\
        TestEncoder,\
        TestGetSetParameters,\
        TestDecoder
from sepia.test.seq.test_transformer import TestTransformer

if __name__ == "__main__":
    
    unittest.main(verbosity=2)
