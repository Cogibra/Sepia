from jax import numpy as jnp
from jax import grad
import jax

import numpy as np
import numpy.random as npr
aa_keys = "arndcqeghilkmfpstwyvuox"
NULL_ELEMENT = "-"

def make_sequence_dict(vocabulary: str, vector_length: int=32, my_seed: int=42) -> dict:

    sequence_dict = {}

    npr.seed(my_seed)

    vocabulary_list = list(vocabulary)
    vocabulary_list.sort()

    sequence_dict[NULL_ELEMENT] = npr.rand(1, vector_length)
    for element in vocabulary:
        sequence_dict[element.lower()] = npr.rand(1, vector_length)


    return sequence_dict

def make_token_dict(vocabulary: str) -> dict:

    token_dict = {}

    vocabulary_list = list(vocabulary)
    vocabulary_list.sort()

    token_dict[NULL_ELEMENT] = np.array(0).reshape(-1,1)
    for token, element in enumerate(vocabulary):
        token_dict[element.lower()] = np.array(token + 1).reshape(1,-1)

    return token_dict

def token_to_one_hot(token: jnp.array, one_hot: jnp.array) -> jnp.array:

    return one_hot.at[token].set(1.0)

vmap_to_one_hots = jax.vmap(token_to_one_hot)

def tokens_to_one_hot(tokens: jnp.array, max_length: int=None) -> jnp.array:

    if max_length is None:
        max_length = jnp.max(tokens)+1

    # pre-allocate one-hot array
    one_hots = jnp.zeros((tokens.shape[0], max_length))

    return vmap_to_one_hots(tokens, one_hots)

def sequence_to_vectors(sequence: str, sequence_dict: dict, pad_to: int=256) -> jnp.array:

    vectors = None

    for element in sequence:

        if vectors is None:
            vectors = sequence_dict[element.lower()]
        else:
            vectors = np.append(vectors, sequence_dict[element.lower()], axis=0)

    while vectors.shape[0] < pad_to:
        vectors = np.append(vectors, sequence_dict[NULL_ELEMENT], axis=0)

    return vectors

def vectors_to_sequence(sequence_vectors: jnp.array, sequence_dict: dict) -> str:

    sequence = ""

    for element in sequence_vectors:

        best_match = float("Inf")
        temp_character = ''

        for key in sequence_dict.keys():

            l1_distance = jnp.sum(jnp.sqrt((element - sequence_dict[key])**2))

            if l1_distance < best_match:

                best_match = 1.0 * l1_distance
                temp_character = key

        sequence += temp_character

    return sequence


