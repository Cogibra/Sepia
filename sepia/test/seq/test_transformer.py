import unittest

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from sepia.seq.transformer import Transformer
from sepia.seq.functional import \
        NICEParametersWB, \
        NICEParametersW, \
        SelfAttentionWB, \
        SelfAttentionW, \
        EncodedAttentionW,\
        EncoderParams, \
        DecoderParams, \
        MLPParams, \
        self_attention, \
        encoder, \
        decoder, \
        bijective_forward, \
        bijective_reverse

class TestTransformer(unittest.TestCase):

    def setUp(self):
        pass

    def test_fit(self):
        model = Transformer(decoder_size=1, encoder_size=1)

        ha_tag = "YPYDVPDYA"

        dataloader = [[ha_tag]]

        model.fit(dataloader, max_steps=2)