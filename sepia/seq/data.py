from jax import numpy as jnp
from jax import grad

import numpy.random as npr
aa_keys = "arndcqeghilkmfpstwyvuox"

def get_sequence_dict(vocabulary: str, vector_length: int=32, my_seed: int=42):

    sequence_dict = {}

    npr.seed(my_seed)

    vocabulary_list = list(vocabulary)
    vocabulary_list.sort()

    for element in vocabulary:
        sequence_dict[element.lower()] = npr.rand(1, vector_length)

    return sequence_dict

def sequence_to_vectors(sequence: str, sequence_dict: dict) -> jnp.array:

    vectors = None

    for element in sequence:

        if vectors == None:
            vectors = sequence_dict[element.lower()]

        else:
            vectors = np.append(vectors, sequence_dict[element.lower()], axis=0)

    return vectors

def vectors_to_sequence(sequence_vectors: jnp.array, sequence_dict: dict) -> str:

    sequence = ""

    for element in sequence_vectors:

        best_match = float("Inf")
        temp_character = ''

        for key in sequence_dict.keys():

            l1_distance = jnp.abs(element - sequence_dict[key])

            if l1_distance < best_match:

                best_match = 1.0 * l1_distance
                temp_character = key

        sequence += temp_character

        if vectors == None:
            vectors = sequence_dict[element.lower()]

        else:
            vectors = np.append(vectors, sequence_dict[element.lower()], axis=0)

    return vectors


