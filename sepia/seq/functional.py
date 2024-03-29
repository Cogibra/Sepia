from functools import reduce

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple

from sepia.seq.data import \
        make_sequence_dict, \
        vectors_to_sequence, \
        sequence_to_vectors

NICEParametersWB = namedtuple("NICEParametersWB", field_names=("weights", "biases"))
NICEParametersW = namedtuple("NICEParametersW", field_names=("weights"))
SelfAttentionWB = namedtuple("SelfAttentionWB", field_names=("weights", "biases"))
SelfAttentionW = namedtuple("SelfAttentionW", field_names=("weights"))
EncodedAttentionW = namedtuple("EncodedAttentionW", \
        field_names=("self_weights", "encoded_weights")) 
MLPParams = namedtuple("MLPParams", \
        field_names=("mlp_weights", "mlp_biases"))
LinearParams = namedtuple("LinearParams", \
        field_names=("linear_weights"))

EncoderLayerParams = namedtuple("EncoderLayerParams", \
        field_names=("attention_weights", "mlp_params"))
DecoderLayerParams = namedtuple("DecoderLayerParams", \
        field_names=("encoded_attention", "mlp_params"))
# Stands for MultiHeadEncoderLayerParams
MHEncoderLayerParams = namedtuple("MHEncoderLayerParams",\
        field_names=("attention_weights", "mlp_params", \
        "linear_weights", "reshape_weights"))
# Stands for MultiHeadDecoderLayerParams
MHDecoderLayerParams = namedtuple("MHDecoderLayerParams",\
        field_names=("encoded_attention", "mlp_params", \
        "linear_weights", "reshape_weights"))

dot = lambda a, b: jnp.dot(a, b.T)
seq_dot = jax.vmap(dot)
batch_seq_dot = jax.vmap(seq_dot)

def make_layers_tuple(depth: int, name: str="weights") -> namedtuple:

    field_names = [f"{name}{ii}" for ii in range(depth)]
    layers_tuple = namedtuple(f"{name}", field_names)

    return layers_tuple

def mlp(x: jnp.array, parameters: MLPParams, activation=jax.nn.relu) -> jnp.array:

    for ii in range(len(parameters.mlp_weights)-1):
        
        x = jnp.matmul(x, parameters.mlp_weights[ii]) 
        x = activation(x + parameters.mlp_biases[ii])

    x = jnp.matmul(x, parameters.mlp_weights[-1]) 
    x = x + parameters.mlp_biases[-1]

    return x

def dot_product_attention(key: jnp.array, query: jnp.array, value: jnp.array) -> jnp.array:

    dim_k = query.shape[-1]
    raw_attention = jnp.matmul(query, key.transpose(0,2,1)) / jnp.sqrt(dim_k)

    attention = jax.nn.softmax(raw_attention, axis=-2)

    output = jnp.matmul(attention, value)

    return output


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

    return dot_product_attention(key, query, value)

multihead_dot_product_attention = jax.pmap(dot_product_attention)
multihead_dot_product_attention_v = jax.vmap(dot_product_attention)
multilinear = jax.vmap(jnp.matmul, in_axes=(None, 0))

def multihead_attention(x: jnp.array, parameters: SelfAttentionW, \
        linear_weights: jnp.array, reshape_weights: jnp.array) -> jnp.array:

    kqv_split = x.shape[-1]
    
    key_query_value = jnp.matmul(x, parameters.weights)

    key = key_query_value[:,:,0:kqv_split]
    query = key_query_value[:,:,kqv_split:2*kqv_split]
    value = key_query_value[:,:,2*kqv_split:3*kqv_split]

    # project key, query, value, to multiple
    keys = multilinear(key, linear_weights[0])
    querys = multilinear(query, linear_weights[1])
    values = multilinear(value, linear_weights[2])

    multi_attention = multihead_dot_product_attention_v(keys, querys, values)

    # concatenate multihead attention
    concatenated = multi_attention[0]
    for ii in range(1, multi_attention.shape[0]):
        concatenated = jnp.append(concatenated, multi_attention[ii], axis=-1)

    reshaped_attention = jnp.matmul(concatenated, reshape_weights)

    return reshaped_attention 

def encoder_layer(x: jnp.array, parameters: EncoderLayerParams) -> jnp.array:

    attention = self_attention(x, parameters.attention_weights)
    attention_residual = x + attention

    # batch by sequence length by token dim
    gain = 1.0
    standard_deviation = jnp.std(attention_residual, axis=(-2,-1), keepdims=True)
    attention_mean = jnp.mean(attention_residual, axis=(-2,-1), keepdims=True)

    normed_attention = (attention_residual - attention_mean) * (gain / standard_deviation)

    output = mlp(normed_attention, parameters.mlp_params)
    output_residual = normed_attention + output
       
    output_standard_deviation = jnp.std(output_residual, axis=(-2,-1), keepdims=True)
    output_mean = jnp.mean(output_residual, axis=(-2,-1), keepdims=True)

    normed_output = (output_residual - output_mean) * (gain / output_standard_deviation)

    return normed_output

def multihead_encoder_layer(x: jnp.array, parameters: MHEncoderLayerParams) -> jnp.array:

    attention = multihead_attention(x, parameters.attention_weights, \
            parameters.linear_weights, parameters.reshape_weights)
    attention_residual = x + attention

    # batch by sequence length by token dim
    gain = 1.0
    standard_deviation = jnp.std(attention_residual, axis=(-2,-1), keepdims=True)
    attention_mean = jnp.mean(attention_residual, axis=(-2,-1), keepdims=True)

    normed_attention = (attention_residual - attention_mean) * (gain / standard_deviation)

    output = mlp(normed_attention, parameters.mlp_params)
    output_residual = normed_attention + output
       
    output_standard_deviation = jnp.std(output_residual, axis=(-2,-1), keepdims=True)
    output_mean = jnp.mean(output_residual, axis=(-2,-1), keepdims=True)

    normed_output = (output_residual - output_mean) * (gain / output_standard_deviation)

    return normed_output

def encoded_attention(x: jnp.array, encoded: jnp.array, parameters: EncodedAttentionW) -> jnp.array:
    # encoded_attention applies dot product self attention 
    # using the key and query vectors from `encoded`
    # and the value vector from `x`

    kqv_split = x.shape[-1]
    
    key_query_value = jnp.matmul(x, parameters.weights)

    value = key_query_value[:,:,2*kqv_split:3*kqv_split]

    encoded_key_query_value = jnp.matmul(encoded, parameters.weights)

    encoded_key = encoded_key_query_value[:,:,0:kqv_split]
    encoded_query = encoded_key_query_value[:,:,kqv_split:2*kqv_split]

    dim_k = encoded_query.shape[-1]

    raw_attention = jnp.matmul(encoded_query, encoded_key.transpose(0,2,1)) / jnp.sqrt(dim_k)

    attention = jax.nn.softmax(raw_attention, axis=-2)

    output = jnp.matmul(attention, value)
    
    return output

def multihead_encoded_attention(x: jnp.array, encoded: jnp.array, parameters: EncodedAttentionW,\
        linear_weights: jnp.array, reshape_weights: jnp.array) -> jnp.array:

    kqv_split = x.shape[-1]
    
    key_query_value = jnp.matmul(x, parameters.weights)
    value = key_query_value[:,:,2*kqv_split:3*kqv_split]

    encoded_key_query_value = jnp.matmul(encoded, parameters.weights)
    encoded_key = encoded_key_query_value[:,:,0:kqv_split]
    encoded_query = encoded_key_query_value[:,:,kqv_split:2*kqv_split]

    encoded_keys = multilinear(encoded_key, linear_weights[0])
    encoded_querys = multilinear(encoded_query, linear_weights[1])
    values = multilinear(value, linear_weights[2])

    dim_k = encoded_querys.shape[-1]

    multi_attention = multihead_dot_product_attention_v(encoded_keys, encoded_querys, values)

    # concatenate multihead attention
    concatenated = multi_attention[0]
    for ii in range(1, multi_attention.shape[0]):
        concatenated = jnp.append(concatenated, multi_attention[ii], axis=-1)

    reshaped_attention = jnp.matmul(concatenated, reshape_weights)

    return reshaped_attention 

def multihead_decoder_layer(x: jnp.array, encoded: jnp.array, parameters: DecoderLayerParams) -> jnp.array:
    # encoded is the last output from encoder layers
    # sometimes called 'memory'

    attention = multihead_attention(x, parameters.encoded_attention.self_weights, \
            parameters.linear_weights[0], parameters.reshape_weights[0])
    attention_residual = x + attention

    # batch by sequence length by token dim
    gain = 1.0
    standard_deviation = jnp.std(attention_residual, axis=(-2,-1), keepdims=True)
    attention_mean = jnp.mean(attention_residual, axis=(-2,-1), keepdims=True)

    normed_attention = (attention_residual - attention_mean) * (gain / standard_deviation)

    my_encoded_attention = multihead_encoded_attention(\
            normed_attention, encoded, parameters.encoded_attention.encoded_weights,\
            parameters.linear_weights[1], parameters.reshape_weights[1])

    encoded_attention_residual = normed_attention + my_encoded_attention

    encoded_standard_deviation = jnp.std(encoded_attention_residual, axis=(-2,-1), keepdims=True)
    encoded_attention_mean = jnp.mean(encoded_attention_residual, axis=(-2,-1), keepdims=True)

    normed_encoded_attention = (encoded_attention_residual - encoded_attention_mean) \
            * (gain / encoded_standard_deviation)

    output = mlp(normed_encoded_attention, parameters.mlp_params)

    output_residual = normed_encoded_attention + output

    output_residual_standard_deviation = jnp.std(output_residual, axis=(-2,-1), keepdims=True)
    output_residual_mean = jnp.mean(output_residual, axis=(-2,-1), keepdims=True)

    normed_output = (output_residual - output_residual_mean) \
            * (gain / output_residual_standard_deviation)

    return normed_output

def decoder_layer(x: jnp.array, encoded: jnp.array, parameters: DecoderLayerParams) -> jnp.array:
    # encoded is the last output from encoder layers
    # sometimes called 'memory'

    attention = self_attention(x, parameters.encoded_attention.self_weights)
    attention_residual = x + attention

    # batch by sequence length by token dim
    gain = 1.0
    standard_deviation = jnp.std(attention_residual, axis=(-2,-1), keepdims=True)
    attention_mean = jnp.mean(attention_residual, axis=(-2,-1), keepdims=True)

    normed_attention = (attention_residual - attention_mean) * (gain / standard_deviation)

    my_encoded_attention = encoded_attention(\
            normed_attention, encoded, parameters.encoded_attention.encoded_weights)

    encoded_attention_residual = normed_attention + my_encoded_attention

    encoded_standard_deviation = jnp.std(encoded_attention_residual, axis=(-2,-1), keepdims=True)
    encoded_attention_mean = jnp.mean(encoded_attention_residual, axis=(-2,-1), keepdims=True)

    normed_encoded_attention = (encoded_attention_residual - encoded_attention_mean) \
            * (gain / encoded_standard_deviation)

    output = mlp(normed_encoded_attention, parameters.mlp_params)

    output_residual = normed_encoded_attention + output

    output_residual_standard_deviation = jnp.std(output_residual, axis=(-2,-1), keepdims=True)
    output_residual_mean = jnp.mean(output_residual, axis=(-2,-1), keepdims=True)

    normed_output = (output_residual - output_residual_mean) \
            * (gain / output_residual_standard_deviation)

    return normed_output

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

batch_bijective_forward = jax.vmap(bijective_forward, in_axes=(0,None))

def bijective_reverse(sequence_features: jnp.array, \
        parameters: NICEParametersWB) -> jnp.array:
    """
    This function implements the reverse pass of a nonlinear neural bijective transform (Faury et al. 2019, Dinh et al. 2014)
    the method is NICE := non-linear indepedent components estimation

    args: 
    sequence_vector represents a biological sequence (DNA, RNA, or amino acid) after being translated to a jax numpy array 
    parameters contains 3 matrices representing 3 matmuls/neural layers and optional biases for each

    usage:

    ```
    # forward pass  
    ha_tag = "YPYDVPDYA"
    sequence_vectors = sequence_to_vectors(ha_tag, aa_sequence_dict)
    tokens = bijective_forward(sequence_vectors, parameters)

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

batch_bijective_reverse = jax.vmap(bijective_reverse, in_axes=(0,None))

def get_parameters(parameters: namedtuple) -> jnp.array:

    np_parameters = None

    for ii, param in enumerate(parameters):
        
        # indirect workaround for testing whether the param is a namedtuple
        if "_field" in dir(param)[35]:
            np_param = get_parameters(param)
        else:
            np_param = param

        np_param = np_param.reshape(-1)
        if np_parameters is None:
            np_parameters = np_param
        else:
            np_parameters = jnp.append(np_parameters, np_param)

    return np_parameters

def set_parameters(np_parameters: jnp.array, parameters: namedtuple) -> namedtuple:

    np_shape = np_parameters.shape
    p_shape = get_parameters(parameters).shape
    exception_message = f"numpy array shape {np_shape} and parameter shape {p_shape} don't match"

    assert np_shape == p_shape, exception_message

    param_start = 0
    new_params = {}
    for ii, param in enumerate(parameters):
        # indirect workaround for testing whether the param is a namedtuple
        if "_field" in dir(param)[35]:
            temp_np_param = get_parameters(param)
            param_stop = param_start + temp_np_param.shape[0]
            np_param = np_parameters[param_start:param_stop]
            new_params[parameters._fields[ii]] = set_parameters(np_param, param)

        else:
            param_stop = param_start + reduce(lambda a,b: a*b, param.shape)
            np_param = np_parameters[param_start:param_stop].reshape(param.shape)
            new_params[parameters._fields[ii]] = np_param

        param_start = param_stop

    new_parameters = type(parameters)(**new_params)

    return new_parameters


