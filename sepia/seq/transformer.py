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
        sequence_to_vectors, \
        batch_sequence_to_vectors

from sepia.seq.dataloader import SeqDataLoader

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
        get_parameters,\
        set_parameters,\
        MLPParams 

# functions
from sepia.seq.functional import \
        encoder, \
        decoder, \
        bijective_forward, \
        batch_bijective_forward, \
        batch_bijective_reverse, \
        bijective_reverse

from sepia.seq.data import \
        make_sequence_dict, \
        vectors_to_sequence, \
        sequence_to_vectors

TransformerParams = namedtuple("TransformerParams", \
        field_names=("token_params", "encoder_params", "decoder_params"))

def cross_entropy(predicted: jnp.array, target: jnp.array, ignore_index: int=None) -> float:

    predicted_softmax = jax.nn.softmax(predicted)

    if ignore_index is not None:
        dont_ignore = 1.0 * (jnp.argmax(target) != ignore_index)
    else: 
        dont_ignore = 1.0

    ce = target * jnp.log(predicted_softmax) * dont_ignore

    assert jnp.mean(ce) <= 0.0, f"c. entropy negative {-jnp.mean(ce)}\n targets: {target.max()}\n pred {predicted_softmax.max()}"

    return - jnp.mean(ce)

def mae(predicted, target):

    return jnp.mean(jnp.sqrt((predicted - target)**2))

def mse(predicted, target):

    return jnp.mean((predicted - target)**2)

def glorot(*args):

    if len(args) > 1:
        dim_in = args[-2]
    else:
        dim_in = args[-1]
    dim_out = args[-1]

    std_dev = jnp.sqrt(2 / (dim_in + dim_out))

    return npr.randn(*args) * std_dev


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
        self.encoder_size = query_kwargs("encoder_size", 4, **kwargs)
        self.decoder_size = query_kwargs("decoder_size", 4, **kwargs)
        self.hidden_dim = 64
        self.mlp_hidden_dim = 48 
        self.mlp_activation = jax.nn.relu
        self.my_seed = 13
        self.init_scale = 1
        self.mask_rate = 0.05
        self.lr = query_kwargs("lr", 3e-3, **kwargs)
        self.loss_fn = cross_entropy

        self.initialize_model()

    def get_token_dict(self) -> dict:

        return self.token_dict

    def get_seq_length(self) -> int:

        return self.seq_length

    def get_token_dim(self) -> int:

        return self.token_dim

    def set_token_dict(self, new_token_dict: dict):

        self.token_dict = new_token_dict

    def set_seq_length(self, new_seq_length: int):

        self.seq_length = new_seq_length

    def set_token_dim(self, new_token_dim):

        self.token_dim = new_token_dim

    def initialize_model(self):

        self.token_dict = make_token_dict(self.vocab)
        
        # tokenizer
        # tokenizer mlp transforms 1/3 of the vector at a time (NICE)
        tokenizer_weight_dim = self.token_dim // 3
        token_weights = glorot(3, tokenizer_weight_dim, tokenizer_weight_dim*2)
        token_parameters = NICEParametersW(weights=token_weights)

        # encoder stack
        encoder_stack = []
        EncoderStack = make_layers_tuple(depth=self.decoder_size, name="decoder")
        for ii in range(self.encoder_size): 
            weights = glorot(self.token_dim, self.encoder_dim)
            attention_weights = SelfAttentionW(weights = weights)

            mlp_weights = [glorot(self.token_dim, self.mlp_hidden_dim), \
                    glorot(self.mlp_hidden_dim, self.mlp_hidden_dim), \
                    glorot(self.mlp_hidden_dim, self.token_dim)]

            mlp_biases = [glorot(self.mlp_hidden_dim), \
                    glorot(self.mlp_hidden_dim), \
                    glorot(self.token_dim)]

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
        encoder_stack = EncoderStack(*encoder_stack)


        # decoder stack
        decoder_stack = []
        DecoderStack = make_layers_tuple(depth=self.decoder_size, name="decoder")
        for ii in range(self.decoder_size): 
            mlp_weights = [glorot(self.token_dim, self.mlp_hidden_dim), \
                    glorot(self.mlp_hidden_dim, self.mlp_hidden_dim), \
                    glorot(self.mlp_hidden_dim, self.token_dim)]

            mlp_biases = [glorot(self.mlp_hidden_dim), \
                    glorot(self.mlp_hidden_dim), \
                    glorot(self.token_dim)]

            mlp_weights_tuple = make_layers_tuple(depth=len(mlp_weights), name="weights")
            mlp_biases_tuple = make_layers_tuple(depth=len(mlp_biases), name="biases")

            mlp_weights = mlp_weights_tuple(*mlp_weights)
            mlp_biases = mlp_biases_tuple(*mlp_biases)

            mlp_params = MLPParams(mlp_weights=mlp_weights,\
                    mlp_biases=mlp_biases)

            # decoder self-attention
            weights_a = glorot(self.token_dim, self.encoder_dim)
            self_attention_weights = SelfAttentionW(weights = weights_a)

            # encoder-decoder attention
            weights_b = glorot(self.token_dim, self.encoder_dim)
            encoded_weights = SelfAttentionW(weights = weights_b)

            attention_weights = EncodedAttentionW(\
                    self_weights=self_attention_weights,\
                    encoded_weights=encoded_weights)

            decoder_parameters = DecoderParams( \
                encoded_attention = attention_weights, \
                mlp_params = mlp_params)

            decoder_stack.append(decoder_parameters)

        decoder_stack = DecoderStack(*decoder_stack)

        self.parameters = TransformerParams(token_parameters, \
                encoder_stack, \
                decoder_stack)

        self.update = optimizer.adam
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
        encoded = x 
        for encoder_params in encoder_stack:

            encoded = encoder(encoded, encoder_params)

        # decoder stack: list of encoder parameters
        decoded = 1.0 * encoded
        for decoder_params in decoder_stack:

            decoded = decoder(decoded, encoded, decoder_params)

        return decoded
    
    def forward_encode(self, x: jnp.array, parameters: tuple) -> jnp.array:
        """
        called after converting string sequences to vectors 
        """

        token_parameters = parameters[0]
        encoder_stack = parameters[1]
        decoder_stack = parameters[2]
        
        # encoder stack: list of encoder parameters
        encoded = x 
        for encoder_params in encoder_stack:

            encoded = encoder(encoded, encoder_params)

        return encoded
    
    def encode(self, sequence: str) -> jnp.array:
        # convert string sequence to numerical vector
        tokens = sequence_to_vectors(sequence, self.token_dict, \
                pad_to = self.seq_length)

        one_hot = tokens_to_one_hot(tokens, pad_to = self.seq_length,\
                pad_classes_to = self.token_dim)

        vector_tokens = bijective_forward(one_hot, self.parameters[0])[None,:,:]
        #vector_tokens = batch_bijective_forward(one_hot, self.parameters[0])
        encoded = self.forward_encode(vector_tokens, self.parameters)

        return encoded


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

        vector_tokens = bijective_forward(one_hot, self.parameters[0])[None,:,:]
        #vector_tokens = batch_bijective_forward(one_hot, self.parameters[0])
        decoded = self.forward(vector_tokens, self.parameters)
        output_tokens = bijective_reverse(decoded[0], self.parameters[0])

        output_sequence = one_hot_to_sequence(output_tokens, self.token_dict)
        return output_sequence

    def calc_loss(self, masked_tokens, target, parameters) -> float:

        decoded = self.forward(masked_tokens, parameters)

        predicted_tokens = batch_bijective_reverse(decoded, parameters[0])

        loss = self.loss_fn(predicted_tokens, target)

        return loss

    def train_step(self, batch: tuple):

        one_hot = batch
        vector_tokens = batch_bijective_forward(one_hot, self.parameters[0])

        vector_mask = 1.0 * (npr.rand(*vector_tokens.shape[:-1],1) > self.mask_rate)

        masked_tokens = vector_tokens * vector_mask 

        grad_loss = grad(self.calc_loss, argnums=2)

        # splitting these roles (returning loss or returning grads) might speed up training
        loss = self.calc_loss(masked_tokens, one_hot, self.parameters)
        my_grad = grad_loss(masked_tokens, one_hot, self.parameters)

        return loss, my_grad

    def fit(self, dataloader, **kwargs) -> None:
        # training loop

        # dataloader is an iterable that returns batches

        max_epochs = query_kwargs("max_epochs", 100, **kwargs)
        display_count = query_kwargs("display_count", 2, **kwargs)
        save_count = query_kwargs("save_count", 0, **kwargs)
        save_every = max_epochs // max([1, save_count])
        display_every = max_epochs // max([1, display_count])

        val_dataloader = query_kwargs("val_dataloader", None, **kwargs)

        for epoch in range(max_epochs):
            
            cumulative_loss = None
            for batch_index, batch in enumerate(dataloader):

                loss, my_grad = self.train_step(batch)

                cumulative_loss = loss if cumulative_loss is None else cumulative_loss+loss  

                

                self.parameters, self.update_info = optimizer.step(\
                        self.parameters, my_grad, lr=self.lr, \
                        update=self.update, info=self.update_info)


            if (display_count and epoch % display_every == 0) or epoch == max_epochs-1:
                if val_dataloader is not None:
                    pass
                print(f"loss at epoch {epoch}:  {cumulative_loss / (batch_index+1.):.3e}")
            if (epoch % save_every == 0 or epoch == max_epochs-1) and save_count:
                checkpoint_path = os.path.join("parameters", f"temp_epoch{epoch}.npy") 
                print(f"saving checkpoint at epoch {epoch} to {checkpoint_path}")
                jnp.save(checkpoint_path, get_parameters(self.parameters))
                checkpoint_path = os.path.join("parameters", f"temp_epoch{epoch}.npy") 
                print(f"saving checkpoint at epoch {epoch} to {checkpoint_path}")
                jnp.save(checkpoint_path, get_parameters(self.parameters))
    
    def restore_parameters(self, np_parameters: jnp.array):

        self.parameters = set_parameters(np_parameters, self.parameters)

    def load_parameters(self, filepath: str):

        np_parameters = jnp.load(filepath)
        self.restore_parameters(np_parameters)



                


if __name__ == "__main__":
    # ad-hoc test for training transformer

    model = Transformer(lr=1e-2)

    ha_tag = "YPYDVPDYA"

    dataset = [ha_tag, ha_tag, ha_tag]

    token_dict = model.get_token_dict()
    seq_length = model.get_seq_length() 
    token_dim = model.get_token_dim() 
    dataloader = SeqDataLoader(token_dict, seq_length, token_dim, dataset=dataset)

    print(ha_tag, "\n", model(ha_tag))

    model.fit(dataloader, max_epochs=200, display_count=20, save_count=2)

    print(ha_tag, "\n", model(ha_tag))
