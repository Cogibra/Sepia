import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple

from sepia.seq.data import \
        get_sequence_dict, \
        vectors_to_sequence, \
        sequence_to_vectors

NICEParametersWB = namedtuple("NICEParametersWB", field_names=("weights", "biases"))
NICEParametersW = namedtuple("NICEParametersW", field_names=("weights"))
SelfAttentionWB = namedtuple("SelfAttentionWB", field_names=("weights", "biases"))
SelfAttentionW = namedtuple("SelfAttentionW", field_names=("weights"))
EncodedAttentionW = namedtuple("EncodedAttentionW", \
        field_names=("self_weights", "encoded_weights")) 
EncoderParams = namedtuple("EncoderParams", \
        field_names=("attention_weights", "mlp_params"))
DecoderParams = namedtuple("DecoderParams", \
        field_names=("encoded_attention", "mlp_params"))
MLPParams = namedtuple("MLPParams", \
        field_names=("mlp_weights", "mlp_biases", "activation"))

dot = lambda a, b: jnp.dot(a, b.T)
seq_dot = jax.vmap(dot)
batch_seq_dot = jax.vmap(seq_dot)



def mlp(x: jnp.array, parameters: MLPParams) -> jnp.array:

    for ii in range(len(parameters.mlp_weights)-1):
        
        x = jnp.matmul(x, parameters.mlp_weights[ii]) 
        x = parameters.activation(x + parameters.mlp_biases[ii])

    x = jnp.matmul(x, parameters.mlp_weights[-1]) 
    x = x + parameters.mlp_biases[-1]

    return x

def self_attention(x: jnp.array, parameters: SelfAttentionW) -> jnp.array:
    """
    This function is a dot product self-attention layer

    args: 
    x is the input vector (jax array) with dimensions n by s by d, where n is the
    batch size, s is the sequence length, and d is the vector dimension
    parameters is a SelfAttentionW named tuple which includes weights. 
    

    returns:
    output: a jax numpy array
    """

    kqv_split = x.shape[-1]
    
    key_query_value = jnp.matmul(x, parameters.weights)

    key = key_query_value[:,:,0:kqv_split]
    query = key_query_value[:,:,kqv_split:2*kqv_split]
    value = key_query_value[:,:,2*kqv_split:3*kqv_split]

    raw_attention = batch_seq_dot(key, query)[:,:,None]

    attention = jax.nn.softmax(raw_attention, axis=0)

    output = attention * value
    
    return output

def encoder(x: jnp.array, parameters: EncoderParams) -> jnp.array:

    attention = self_attention(x, parameters.attention_weights)

    output = mlp(attention, parameters.mlp_params)
        
    return output

def encoded_attention(x: jnp.array, encoded: jnp.array, parameters: EncodedAttentionW) -> jnp.array:
    # EncDecAttentionW = namedtuple("EncDecAttention", \
    #    field_names=("self_weights", "encoded_weights")) 

    my_self_attention = self_attention(x, parameters.self_weights)
    encoded_attention = self_attention(encoded, parameters.encoded_weights)

    output = my_self_attention + encoded_attention

    return output

def decoder(x: jnp.array, encoded: jnp.array, parameters: DecoderParams) -> jnp.array:

    attention = encoded_attention(x, encoded, parameters.encoded_attention)

    output = mlp(attention, parameters.mlp_params)

    return output

def bijective_forward(sequence_vectors: jnp.array, \
        parameters: NICEParametersWB, pad_to: int=1024) -> jnp.array:
    """
    This function implements the (reversible) forward pass as described in Faury et al. 2019 and Dinh et al. 2014

    args:
    sequence_vector represents a biological sequence (DNA, RNA, or amino acid) after being translated to a jax numpy array
    parameters contains 3 matrices representing 3 matmuls/neural layers and optional biases for each
    pad_to is an integer specifying the final length of the sequence. If pad_to is less than len(sequence), the sequence will be truncated

    usage:

    ```
    # forward pass  
    my_seq = "wyavilmf"
    sequence_vectors = sequence_to_vectors(my_seq, sequence_dict)
    tokens = bijective_forward(sequence_vectors, parameters, pad_to=2048)

    # reverse pass 
    detokens = bijective_reverse(tokens, parameters)
    # now the tokens should be close to the sequence_dict items
    result_sequence = vectors_to_sequence(detokens, sequence_dict)
    ```

    What's happening? 

    # forward
    v_1 = u_1
    v_2 = u_2 + f(u_1)

    # reverse
    u_1 = v_1
    u_2 = v_2 - f(v_1)

    where f() is a non-linear transformation, u_n are input components and v_n
    are the outputs of the forward operation. 

    as described in Faury et al., the forward and reverse passes are arranged 
    to retain some part of the inputs at each stage, e.g. a masked portion of 
    an input vector is passed unchanged while the rest is subject to non-linear
    transformation. By stacking 3 of these transformations (we use dense NN layers)
    with complementary identity masks, the characteristics of non-linearity can be
    preserved while at the same time enabling reversibility.

    """
    
    vector_length = sequence_vectors.shape[-1] 
    mask_length = vector_length // 3 
    activation = jnp.tanh

    u_1 = sequence_vectors[:, :mask_length]
    u_2 = sequence_vectors[:, mask_length:]

    v_10 = u_1 
    v_20 = u_2 + activation(jnp.matmul(u_1, parameters.weights[0]))
    
    v = jnp.append(v_10, v_20, axis=-1)

    v_11 = v[:, mask_length:-mask_length] 
    v_21 = jnp.append(v[:, :mask_length], v[:, -mask_length:], axis=-1) 

    w_10 = v_11
    w_20 = v_21 + activation(jnp.matmul(v_11, parameters.weights[1]))

    w = jnp.append(jnp.append(w_20[:, :mask_length], w_10, axis=-1), \
            w_20[:, -mask_length:], axis=-1)

    w_11 = w[:, -mask_length:]
    w_21 = w[:, :-mask_length]

    x_10 = w_11
    x_20 = w_21 + activation(jnp.matmul(w_11, parameters.weights[2]))

    tokens = jnp.append(x_20, x_10, axis=-1)
    
    return tokens

def bijective_reverse(sequence_features: jnp.array, \
        parameters: NICEParametersWB) -> jnp.array:
    """
    This function implements the reverse pass of a nonlinear neural bijective transform (Faury et al. 2019, Dinh et al. 2014)
    the method is NICE := non-linear indepedent components estimation

    args: 
    sequence_vector represents a biological sequence (DNA, RNA, or amino acid) after being translated to a jax numpy array 
    parameters contains 3 matrices representing 3 matmuls/neural layers and optional biases for each
    pad_to is an integer specifying the final length of the sequence. If pad_to is less than len(sequence), the sequence will be truncated

    usage:

    ```
    # forward pass  
    ha_tag = "YPYDVPDYA"
    sequence_vectors = sequence_to_vectors(ha_tag, aa_sequence_dict)
    tokens = bijective_forward(sequence_vectors, parameters, pad_to=2048)

    # reverse pass 
    detokens = bijective_reverse(tokens, parameters)
    # now the tokens should be close to the sequence_dict items
    result_sequence = vectors_to_sequence(detokens, sequence_dict)
    ```

    # reverse
    u_1 = v_1
    u_2 = v_2 - f(v_1)

    """

    vector_length = sequence_features.shape[-1] 
    mask_length = vector_length // 3 
    activation = jnp.tanh

    x_13 = sequence_features[:, -mask_length:]
    x_23 = sequence_features[:, :-mask_length]

    w_12 = x_13
    w_22 = x_23 - activation(jnp.matmul(x_13, parameters.weights[2])) 

    w = jnp.append(w_22, w_12, axis=-1)

    w_13 = w[:, mask_length:-mask_length] 
    w_23 = jnp.append(w[:, :mask_length], w[:, -mask_length:], axis=-1) 

    v_12 = w_13
    v_22 = w_23 - activation(jnp.matmul(w_13, parameters.weights[1]))

    v = jnp.append(jnp.append(v_22[:, :mask_length], v_12, axis=-1), \
            v_22[:, -mask_length:], axis=-1)

    v_13 = v[:, :mask_length]
    v_23 = v[:, mask_length:]

    u_12 = v_13
    u_22 = v_23 - activation(jnp.matmul(v_13, parameters.weights[0]))

    sequence_vectors = jnp.append(u_12, u_22, axis=-1)

    return sequence_vectors
