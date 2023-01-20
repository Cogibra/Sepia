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

from sepia.seq.dataloader import SeqDataLoader

class TestTransformer(unittest.TestCase):

    def setUp(self):
        pass

    def test_fit(self):
        model = Transformer(decoder_size=1, encoder_size=1)

        ha_tag = "YPYDVPDYA"

        dataset = [ha_tag]

        token_dict = model.get_token_dict()
        seq_length = model.get_seq_length()
        token_dim = model.get_token_dim()

        dataloader = SeqDataLoader(token_dict, seq_length, token_dim, dataset=dataset, batch_size=1)

        model.fit(dataloader, max_epochs=2)
