import argparse
import os
import copy

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple

from sepia.common import query_kwargs
import sepia.optimizer as optimizer

from sepia.seq.data import \
        aa_keys, \
        make_sequence_dict, \
        make_token_dict, \
        tokens_to_one_hot, \
        compose_batch_tokens_to_one_hot, \
        one_hot_to_sequence, \
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
        make_layers_tuple, \
        MLPParams 

# functions
from sepia.seq.functional import \
        encoder, \
        decoder, \
        bijective_forward, \
        bijective_reverse

from sepia.seq.data import \
        make_sequence_dict, \
        vectors_to_sequence, \
        sequence_to_vectors

TransformerParams = namedtuple("TransformerParams", \
        field_names=("token_params", "encoder_params", "decoder_params"))

def cross_entropy(predicted: jnp.array, target: jnp.array) -> float:

    predicted_softmax = jax.nn.softmax(predicted)
    ce = target * jnp.log(predicted_softmax)\

    return - jnp.mean(ce)

def mae(predicted, target):

    return jnp.mean(jnp.sqrt((predicted - target)**2))

def mse(predicted, target):

    return jnp.mean((predicted - target)**2)

class Transformer():

    def __init__(self, **kwargs):
        pass
        """
        tokenizer, encoders, decoders
        """

        # architectural dims and details 
        self.vocab = query_kwargs("vocab", aa_keys, **kwargs)
        token_dim = len(self.vocab)
        if token_dim % 3:
            token_dim = token_dim + (3-(token_dim % 3))
        self.token_dim = query_kwargs("token_dim", token_dim, **kwargs)
        self.encoder_dim = self.token_dim * 3

        # expose these to user
        self.seq_length = query_kwargs("seq_length", 10, **kwargs)
        self.encoder_size = query_kwargs("encoder_size", 1, **kwargs)
        self.decoder_size = query_kwargs("decoder_size", 1, **kwargs)
        self.hidden_dim = 64
        self.mlp_hidden_dim = 48 
        self.mlp_activation = jax.nn.relu
        self.my_seed = 13
        self.init_scale = 1e-1
        self.mask_rate = 0.05
        self.lr = query_kwargs("lr", 1e-2, **kwargs)
        self.loss_fn = cross_entropy

        self.initialize_model()

    def initialize_model(self):

        self.token_dict = make_token_dict(self.vocab)
        
        # tokenizer
        # tokenizer mlp transforms 1/3 of the vector at a time (NICE)
        tokenizer_weight_dim = self.token_dim // 3
        token_weights = npr.randn(3, tokenizer_weight_dim, tokenizer_weight_dim*2)
        self.token_parameters = NICEParametersW(weights=token_weights)

        # encoder stack
        encoder_stack = []
        EncoderStack = make_layers_tuple(depth=self.decoder_size, name="decoder")
        for ii in range(self.encoder_size): 
            weights = npr.randn(self.token_dim, self.encoder_dim)*self.init_scale
            attention_weights = SelfAttentionW(weights = weights)

            mlp_weights = [npr.randn(self.token_dim, self.mlp_hidden_dim)*self.init_scale, \
                    npr.randn(self.mlp_hidden_dim, self.mlp_hidden_dim)*self.init_scale, \
                    npr.randn(self.mlp_hidden_dim, self.token_dim)*self.init_scale]

            mlp_biases = [npr.randn(self.mlp_hidden_dim,)*self.init_scale, \
                    npr.randn(self.mlp_hidden_dim,)*self.init_scale, \
                    npr.randn(self.token_dim,)*self.init_scale]

            mlp_weights_tuple = make_layers_tuple(depth=len(mlp_weights), name="weights")
            mlp_biases_tuple = make_layers_tuple(depth=len(mlp_biases), name="biases")

            mlp_weights = mlp_weights_tuple(*mlp_weights)
            mlp_biases = mlp_biases_tuple(*mlp_biases)

            mlp_params = MLPParams(mlp_weights=mlp_weights,\
                    mlp_biases=mlp_biases)

            encoder_parameters = EncoderParams( \
                    attention_weights = attention_weights, \
                    mlp_params = mlp_params)

            encoder_stack.append(encoder_parameters)
        self.encoder_stack = EncoderStack(*encoder_stack)


        # decoder stack
        decoder_stack = []
        DecoderStack = make_layers_tuple(depth=self.decoder_size, name="decoder")
        for ii in range(self.decoder_size): 
            mlp_weights = [npr.randn(self.token_dim, self.mlp_hidden_dim)*self.init_scale, \
                    npr.randn(self.mlp_hidden_dim, self.mlp_hidden_dim)*self.init_scale, \
                    npr.randn(self.mlp_hidden_dim, self.token_dim)*self.init_scale]

            mlp_biases = [npr.randn(self.mlp_hidden_dim,)*self.init_scale, \
                    npr.randn(self.mlp_hidden_dim,)*self.init_scale, \
                    npr.randn(self.token_dim,)*self.init_scale]

            mlp_weights_tuple = make_layers_tuple(depth=len(mlp_weights), name="weights")
            mlp_biases_tuple = make_layers_tuple(depth=len(mlp_biases), name="biases")

            mlp_weights = mlp_weights_tuple(*mlp_weights)
            mlp_biases = mlp_biases_tuple(*mlp_biases)

            mlp_params = MLPParams(mlp_weights=mlp_weights,\
                    mlp_biases=mlp_biases)

            # decoder self-attention
            weights_a = npr.randn(self.token_dim, self.encoder_dim)*self.init_scale
            self_attention_weights = SelfAttentionW(weights = weights_a)

            # encoder-decoder attention
            weights_b = npr.randn(self.token_dim, self.encoder_dim)*self.init_scale
            encoded_weights = SelfAttentionW(weights = weights_b)

            attention_weights = EncodedAttentionW(\
                    self_weights=self_attention_weights,\
                    encoded_weights=encoded_weights)

            decoder_parameters = DecoderParams( \
                encoded_attention = attention_weights, \
                mlp_params = mlp_params)

            decoder_stack.append(decoder_parameters)

        self.decoder_stack = DecoderStack(*decoder_stack)

        self.parameters = TransformerParams(self.token_parameters, \
                self.encoder_stack, \
                self.decoder_stack)

        self.update = optimizer.sgd
        self.update_info = None

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

        return decoded

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
        tokens = sequence_to_vectors(sequence, self.token_dict, \
                pad_to = self.seq_length)

        one_hot = tokens_to_one_hot(tokens, pad_to = self.seq_length,\
                pad_classes_to = self.token_dim)

        parameters = TransformerParams(self.token_parameters, \
                self.encoder_stack, \
                self.decoder_stack)

        vector_tokens = bijective_forward(one_hot, self.token_parameters)[None,:,:]
        decoded = self.forward(one_hot[None,:,:], self.parameters)
        output_tokens = bijective_reverse(decoded[0], \
                self.token_parameters)

        output_sequence = one_hot_to_sequence(decoded[0], self.token_dict)
        print(jnp.argmax(decoded[0], axis=-1))
        print(jnp.argmax(one_hot, axis=-1))

        # use vmap for multiple sequ/biaseence at once?

        return output_sequence

    def calc_loss(self, masked_tokens, target, parameters) -> float:

        
        decoded = self.forward(masked_tokens[None,:,:], parameters)
        predicted_tokens = bijective_reverse(decoded[0], \
                parameters[0])

#        print("calc_loss decoded, masked")
#        print(jnp.argmax(decoded[0], axis=-1))
#        print(jnp.argmax(masked_tokens, axis=-1))
        loss = self.loss_fn(decoded[0], target)

        return loss

    def train_step(self, batch: tuple):

        sequence = batch[0]

        tokens = sequence_to_vectors(sequence, self.token_dict, \
                pad_to = self.seq_length)
        one_hot = tokens_to_one_hot(tokens, pad_to = self.seq_length,\
                pad_classes_to = self.token_dim)

        vector_tokens = bijective_forward(one_hot, self.token_parameters)[None,:,:]

        masked_tokens = one_hot #\
        #        * (npr.rand(*one_hot.shape[0:1],1) > self.mask_rate)

        grad_loss = grad(self.calc_loss, argnums=2)

        # splitting these roles (returning loss or returning grads) might speed up training
#        print("train_step masked, one_hot")
#        print(jnp.argmax(masked_tokens, axis=-1))
#        print(jnp.argmax(one_hot, axis=-1))
        loss = self.calc_loss(masked_tokens, one_hot, self.parameters)
        my_grad = grad_loss(masked_tokens, one_hot, self.parameters)


        return loss, my_grad

    def fit(self, dataloader, **kwargs) -> None:
        # training loop

        # dataloader is an iterable that returns batches

        max_steps = query_kwargs("max_steps", 100, **kwargs)
        display_count = query_kwargs("display_count", 2, **kwargs)
        display_every = max_steps // display_count

        #starting_params = copy.deepcopy(self.parameters) 
        
        for step in range(max_steps):
            
            cumulative_loss = None
            for batch_index, batch in enumerate(dataloader):

                loss, my_grad = self.train_step(batch)

                cumulative_loss = loss if cumulative_loss is None else cumulative_loss+loss  

                

                self.parameters, self.update_info = optimizer.step(\
                        self.parameters, my_grad, lr=self.lr, \
                        update=self.update, info=self.update_info)


            if step % display_every == 0 or step == max_steps-1:
                print(batch[0].lower())
                print(self(batch[0]))
                print(f"loss at step {step}:  {cumulative_loss / (batch_index+1.):.3e}")

                


if __name__ == "__main__":

    model = Transformer()

    ha_tag = "YPYDVPDYA"

    dataloader = [[ha_tag]]

    print(model(ha_tag))

    model.fit(dataloader, max_steps=2000, display_count=20)

    print(model(ha_tag))

    #fit(self, dataloader, **kwargs) -> None:

