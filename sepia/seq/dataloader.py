import argparse
import os
import copy

import jax
from jax import numpy as jnp
from jax import grad

import numpy as np
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
        sequence_to_vectors,\
        batch_sequence_to_vectors

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
        encoder_layer, \
        decoder_layer, \
        bijective_forward, \
        bijective_reverse

from sepia.seq.data import \
        make_sequence_dict, \
        vectors_to_sequence, \
        sequence_to_vectors


class SeqDataLoader():

    def __init__(self, token_dict: dict, seq_length: int, token_dim: int,\
            **kwargs: dict):

        self.token_dict = token_dict
        self.seq_length = seq_length
        self.token_dim = token_dim

        self.shuffle = query_kwargs("shuffle", False, **kwargs)
        self.batch_size = query_kwargs("batch_size", 8, **kwargs)
        self.my_seed = query_kwargs("seed", 13, **kwargs)

        if "dataset" in kwargs.keys():
            self.setup_dataset(kwargs["dataset"])

    def setup_dataset(self, dataset: np.array):
        # shape dataset and convert to one hot vectors
        # dataset is expected to be a 1D array of string sequences

        if type(dataset) == list:
            dataset = np.array(dataset)

        if self.shuffle:
            pass

        remainder = dataset.shape[0] % self.batch_size

        while remainder:

            if remainder:
                append_index = self.batch_size - remainder
                dataset = np.append(dataset, dataset[0:append_index], axis=0)

            remainder = dataset.shape[0] % self.batch_size

        dataset = dataset.reshape(-1)

        token_dataset = batch_sequence_to_vectors(dataset, self.token_dict,\
                pad_to = self.seq_length)

        batch_to_one_hot = compose_batch_tokens_to_one_hot(\
                pad_to = self.seq_length, pad_classes_to = self.token_dim)
        one_hot_dataset = batch_to_one_hot(token_dataset)

        self.dataset = one_hot_dataset.reshape(-1, self.batch_size, \
                self.seq_length, self.token_dim)

    def set_dataset(self, dataset: jnp.array):

        remainder = dataset.shape[1] % self.batch_size

        while remainder:

            if remainder:
                append_index = self.batch_size - remainder
                dataset = np.append(dataset, dataset[0:append_index], axis=0)

            remainder = dataset.shape[0] % self.batch_size


        assert self.seq_length == dataset.shape[-2], f"seq_length {self.seq_length} != {dataset.shape[-2]}"
        assert self.token_dim == dataset.shape[-1], f"token_dim {self.token_dim} != {dataset.shape[-1]}"

        self.dataset = dataset.reshape(-1, self.batch_size, \
                self.seq_length, self.token_dim)

    def save_dataset(self, filepath: str=None):

        if filepath is None:
            filepath = os.path.join("data", "temp.npy")

        jnp.save(filepath, self.dataset)

    def load_dataset(self, filepath: str=None):

        if filepath is None:
            filepath = os.path.join("data", "temp.npy")

        if os.path.exists(filepath):
            self.set_dataset(jnp.load(filepath))
        else:
            print(f"warning, {filepath} does not exist")

    def __len__(self) -> int:

        return len(self.dataset)

    def __getitem__(self, index) -> jnp.array:

        return self.dataset[index:index+1]

    def __iter__(self):

        if self.shuffle:
            pass

        return iter(self.dataset)




