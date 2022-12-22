import unittest

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

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

class TestSelfAttention(unittest.TestCase):
    
    def setUp(self):

        weights = npr.randn(12, 36)
        self.x = npr.randn(16,32,12)

        self.parameters = SelfAttentionW(weights = weights)

    def test_self_attention(self):

        attention = self_attention(self.x, self.parameters)

        self.assertEqual(self.x.shape, attention.shape)
"""
EncodedAttentionW = namedtuple("EncodedAttentionW", \
        field_names=("self_weights", "encoded_weights")) 
EncoderParams = namedtuple("EncoderParams", \
        field_names=("attention_weights", "mlp_params"))
DecoderParams = namedtuple("DecoderParams", \
        field_names=("encoded_attention", "mlp_params"))
MLPParams = namedtuple("MLPParams", \
        field_names=("mlp_weights", "mlp_biases", "activation"))
"""

class TestDecoder(unittest.TestCase):

    def setUp(self):

        weights_a = npr.randn(12, 36)
        self_attention_weights = SelfAttentionW(weights = weights_a)

        weights_b = npr.randn(12, 36)
        encoded_weights = SelfAttentionW(weights = weights_b)

        attention_weights = EncodedAttentionW(\
                self_weights=self_attention_weights,\
                encoded_weights=encoded_weights)

        mlp_weights = [npr.randn(12, 32), npr.randn(32,8), npr.randn(8,12)]
        mlp_biases = [npr.randn(32,), npr.randn(8,), npr.randn(12,)]
        mlp_activation = jax.nn.relu

        mlp_params = MLPParams(mlp_weights=mlp_weights,\
                mlp_biases=mlp_biases,\
                activation=mlp_activation)

        self.decoder_parameters = DecoderParams( \
                encoded_attention = attention_weights, \
                mlp_params = mlp_params)
                
        self.x = npr.randn(16,32,12)
        self.encoded = npr.randn(16,32,12)


    def test_decoder(self):

        encoded = decoder(self.x, self.encoded, self.decoder_parameters)

        self.assertEqual(self.x.shape, encoded.shape)

class TestEncoder(unittest.TestCase):
    
    def setUp(self):

        weights = npr.randn(12, 36)
        self.x = npr.randn(16,32,12)

        attention_weights = SelfAttentionW(weights = weights)

        mlp_weights = [npr.randn(12, 32), npr.randn(32,8), npr.randn(8,12)]
        mlp_biases = [npr.randn(32,), npr.randn(8,), npr.randn(12,)]
        mlp_activation = jax.nn.relu

        mlp_params = MLPParams(mlp_weights=mlp_weights,\
                mlp_biases=mlp_biases,\
                activation=mlp_activation)

        self.parameters = EncoderParams( \
                attention_weights = attention_weights, \
                mlp_params = mlp_params)

    def test_encoder(self):

        encoded = encoder(self.x, self.parameters)

        self.assertEqual(self.x.shape, encoded.shape)

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
