from jax import numpy as jnp
from jax import grad

import numpy as np
import numpy.random as npr
aa_keys = "arndcqeghilkmfpstwyvuox"
NULL_ELEMENT = "-"

def get_sequence_dict(vocabulary: str, vector_length: int=32, my_seed: int=42):

    sequence_dict = {}

    npr.seed(my_seed)

    vocabulary_list = list(vocabulary)
    vocabulary_list.sort()

    sequence_dict[NULL_ELEMENT] = npr.rand(1, vector_length)
    for element in vocabulary:
        sequence_dict[element.lower()] = npr.rand(1, vector_length)


    return sequence_dict

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


