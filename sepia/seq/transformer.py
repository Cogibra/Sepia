import argparse
import os

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple

from sepia.common import query_kwargs

from sepia.seq.data import \
        aa_keys, \
        get_sequence_dict, \
        vectors_to_sequence, \
        sequence_to_vectors

# parameters (namedtuples)
from sepia.seq.functional import \
        NICEParametersWB, \
        NICEParametersW, \
        SelfAttentionWB, \
        SelfAttentionW, \
        EncodedAttentionW, \
        EncoderParams, \
        DecoderParams, \
        MLPParams 

# functions
from sepia.seq.functional import \
        encoder, \
        decoder, \
        bijective_forward, \
        bijective_reverse

from sepia.seq.data import \
        get_sequence_dict, \
        vectors_to_sequence, \
        sequence_to_vectors

class Transformer():

    def __init__(self, **kwargs):
        pass
        """
        tokenizer, encoders, decoders
        """

        # architectural dims and details 
        self.vocab = query_kwargs("vocab", aa_keys, **kwargs)
        self.token_dim = query_kwargs("token_dim", 48, **kwargs)
        self.encoder_dim = self.token_dim * 3

        # expose these to user
        self.encoder_size = 1
        self.decoder_size = 1
        self.hidden_dim = 64
        self.seq_length = 128
        self.mlp_hidden_dim = 48 
        self.mlp_activation = jax.nn.relu
        self.my_seed = 13
        self.init_scale = 1e-3

        self.initialize_model()

    def initialize_model(self):

        self.sequence_dict = get_sequence_dict(self.vocab, self.token_dim, my_seed=self.my_seed)
        
        # tokenizer
        # tokenizer mlp transforms 1/3 of the vector at a time (NICE)
        tokenizer_weight_dim = self.token_dim // 3
        token_weights = npr.randn(3, tokenizer_weight_dim, tokenizer_weight_dim*2)
        self.token_parameters = NICEParametersW(weights=token_weights)

        # encoder stack
        self.encoder_stack = []
        for ii in range(self.encoder_size): 
            weights = npr.randn(self.token_dim, self.encoder_dim)
            attention_weights = SelfAttentionW(weights = weights)

            mlp_weights = [npr.randn(self.token_dim, self.mlp_hidden_dim), \
                    npr.randn(self.mlp_hidden_dim, self.mlp_hidden_dim), \
                    npr.randn(self.mlp_hidden_dim, self.token_dim)]

            mlp_biases = [npr.randn(self.mlp_hidden_dim,), \
                    npr.randn(self.mlp_hidden_dim,), \
                    npr.randn(self.token_dim,)]


            mlp_params = MLPParams(mlp_weights=mlp_weights,\
                    mlp_biases=mlp_biases)

            encoder_parameters = EncoderParams( \
                    attention_weights = attention_weights, \
                    mlp_params = mlp_params)

            self.encoder_stack.append(encoder_parameters)


        # decoder stack
        self.decoder_stack = []
        for ii in range(self.decoder_size): 
            mlp_weights = [npr.randn(self.token_dim, self.mlp_hidden_dim), \
                    npr.randn(self.mlp_hidden_dim, self.mlp_hidden_dim), \
                    npr.randn(self.mlp_hidden_dim, self.token_dim)]

            mlp_biases = [npr.randn(self.mlp_hidden_dim,), \
                    npr.randn(self.mlp_hidden_dim,), \
                    npr.randn(self.token_dim,)]

            mlp_params = MLPParams(mlp_weights=mlp_weights,\
                    mlp_biases=mlp_biases)

            # decoder self-attention
            weights_a = npr.randn(self.token_dim, self.encoder_dim)
            self_attention_weights = SelfAttentionW(weights = weights_a)

            # encoder-decoder attention
            weights_b = npr.randn(self.token_dim, self.encoder_dim)
            encoded_weights = SelfAttentionW(weights = weights_b)

            attention_weights = EncodedAttentionW(\
                    self_weights=self_attention_weights,\
                    encoded_weights=encoded_weights)

            decoder_parameters = DecoderParams( \
                encoded_attention = attention_weights, \
                mlp_params = mlp_params)

            self.decoder_stack.append(decoder_parameters)

    def forward(self, x: jnp.array, parameters: tuple) -> jnp.array:
        """
        numerical pass
        called after converting string sequences to vectors 
        """

        token_parameters = parameters[0]
        encoder_stack = parameters[1]
        decoder_stack = parameters[2]
        
        # encoder stack: list of encoder parameters
        encoded = x #tokens
        for encoder_params in encoder_stack:

            encoded = encoder(encoded, encoder_params)

        # encoder stack: list of encoder parameters
        decoded = 1.0 * encoded
        for decoder_params in decoder_stack:

            decoded = decoder(decoded, encoded, decoder_params)

        output_tokens = bijective_reverse(decoded[0], \
                token_parameters)

        return output_tokens

    def __call__(self, sequence: str) -> str:
        """
        # forward pass  
        my_seq = "wyavilmf"
        sequence_vectors = sequence_to_vectors(my_seq, sequence_dict)
        tokens = bijective_forward(sequence_vectors, parameters)

        # reverse pass 
        detokens = bijective_reverse(tokens, parameters)
        # now the tokens should be close to the sequence_dict items
        result_sequence = vectors_to_sequence(detokens, sequence_dict)
        """
        # forward pass

        # convert string sequence to numerical vector
        vector = sequence_to_vectors(sequence, self.sequence_dict, \
                pad_to = self.seq_length)

        parameters = (self.token_parameters, \
                self.encoder_stack, \
                self.decoder_stack)

        input_tokens = bijective_forward(vector, self.token_parameters)[None,:,:]
        output_tokens = self.forward(input_tokens, parameters)
        output_sequence = vectors_to_sequence(output_tokens, self.sequence_dict)

        # use vmap for multiple sequence at once?

        return output_sequence

    def get_loss(self, masked_tokens, target, parameters) -> float:
        
        predicted_tokens = self.forward(masked_tokens, parameters)

        loss = jnp.mean(jnp.sqrt((predicted_tokens - target)**2))

        return loss

    def train_step(self, batch: tuple):

        sequence = batch[0]

        vector = sequence_to_vectors(sequence, self.sequence_dict, \
                pad_to = self.seq_length)

        input_tokens = bijective_forward(vector, self.token_parameters)[None,:,:]

        parameters = (self.token_parameters, \
                self.encoder_stack, \
                self.decoder_stack)

        self.mask_rate = 0.2

        masked_tokens = input_tokens \
                * (npr.rand(*input_tokens.shape[:2],1) > self.mask_rate)

        grad_loss = grad(self.get_loss, argnums=2)
        loss = self.get_loss(masked_tokens, input_tokens, parameters)

        my_grad = grad_loss(masked_tokens, input_tokens, parameters)

        import pdb; pdb.set_trace()


    def fit(self, **kwargs):
        # training loop

        max_steps = 10
        for step in range(max_steps):
            pass


if __name__ == "__main__":

    pass
