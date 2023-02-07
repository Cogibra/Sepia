from jax import numpy as jnp
from jax import grad
import jax

import numpy as np
import numpy.random as npr
aa_keys = "arndcqeghilkmfpstwyvuox"
NULL_ELEMENT = None

def make_sequence_dict(vocabulary: str, vector_length: int=32, my_seed: int=42) -> dict:

    sequence_dict = {}

    npr.seed(my_seed)

    vocabulary_list = list(vocabulary)
    vocabulary_list.sort()

    sequence_dict[NULL_ELEMENT] = npr.rand(1, vector_length)
    for element in vocabulary:
        sequence_dict[element] = npr.rand(1, vector_length)


    return sequence_dict

def make_token_dict(vocabulary: str) -> dict:

    token_dict = {}

    vocabulary_list = list(vocabulary)
    vocabulary_list.sort()

    token_dict[NULL_ELEMENT] = np.array(0).reshape(-1,1)
    for token, element in enumerate(vocabulary):
        token_dict[element] = np.array(token + 1).reshape(1,-1)

    return token_dict

def token_to_one_hot(token: jnp.array, one_hot: jnp.array) -> jnp.array:

    return one_hot.at[token].set(1.0)

vmap_to_one_hot = jax.vmap(token_to_one_hot)

def tokens_to_one_hot(tokens: jnp.array, pad_classes_to: int=None, \
        pad_to: int=None) -> jnp.array:

    if pad_to is None:
        pad_to = tokens.shape[0]
    if pad_classes_to is None:
        pad_classes_to = jnp.max(tokens)+1

    # pre-allocate one-hot array
    one_hot = jnp.zeros((pad_to, pad_classes_to))

    return vmap_to_one_hot(tokens, one_hot)


def compose_batch_tokens_to_one_hot(\
        pad_to: int, pad_classes_to: int) -> type(lambda x: x):
    """
    Produices a loop-based batch function for converting integer/index tokens
    to a one hot encoding. 
    
    batch_tokens_to_one_hot = jax.vmap(tokens_to_one_hot, in_axes=(0, None, None))
    """

    one_hot = jnp.zeros((pad_to,  pad_classes_to)) 
    def batch_tokens_to_one_hot(tokens):

        return vmap_to_one_hot(tokens, one_hot)
    
    return jax.vmap(batch_tokens_to_one_hot)


def sequence_to_vectors(sequence: str, sequence_dict: dict, pad_to: int=64) -> jnp.array:

    vectors = None

    for element in sequence:

        if vectors is None:
            vectors = sequence_dict[element]
        else:
            vectors = np.append(vectors, sequence_dict[element], axis=0)

    while vectors.shape[0] < pad_to:
        vectors = np.append(vectors, sequence_dict[NULL_ELEMENT], axis=0)

    return vectors

def batch_sequence_to_vectors(batch: tuple, sequence_dict, pad_to: int=64) -> jnp.array:

    vectors = None
    for my_item in batch:

        if vectors is None:
            vectors = sequence_to_vectors(my_item, sequence_dict, pad_to)[None,:,:]
        else:
            vector = sequence_to_vectors(my_item, sequence_dict, pad_to)[None,:,:]
            vectors = jnp.append(vectors, vector, axis=0) 
    return vectors

def one_hot_to_sequence(one_hot: jnp.array, sequence_dict: dict) -> str:

    sequence = ""

    key_dict = {sequence_dict[key].item(): key for key in sequence_dict.keys()}
    for element in one_hot:

        index = jnp.argmax(element).item()
        if index in key_dict.keys():
            sequence += key_dict[index] if index != 0 else ""
        else:
            print(index, "key not in dict")
            sequence += "" #key_dict[list(key_dict.keys())[0]]

    return sequence

def batch_one_hot_to_sequence(one_hot: jnp.array, sequence_dict: dict) -> list:
    """
    process a batch of sequences represented by one hot encodings
    returns a list of strings
    """

    shape_length = len(one_hot.shape)
    assert shape_length == 3, f"expected 3 dims in one_hot batch vector, got {shape_length}"

    output = []
    #loop through the 
    for ii in range(one_hot.shape[0]):
        
        output.append(one_hot_to_sequence(one_hot[ii], sequence_dict))

    return output


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


