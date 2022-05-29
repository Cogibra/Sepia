from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple

GraphBatchTuple = namedtuple("BatchTuple", ("node_features",\
        "edges", "edge_features", "batch_indices"))

def parse_edges_to_adjacency(edges: jnp.array, normalize: bool=True) -> jnp.array:
    """
    parse an array of edges and build and adjacency matrix

    edges is a Nx2 array of sending_node, receiving node. 
    Note that this function produces a bidirectional adjacency matrix 

    """

    max_nodes = jnp.max(jnp.array(edges)) + 1

    #adj_matrix = jnp.zeros((max_nodes, max_nodes))

    adj_matrix = jnp.array([[1 if [ii, jj] in edges else 0 \
            for ii in range(max_nodes)] \
            for jj in range(max_nodes)])

    if 0: #for edge_0, edge_1 in edges:
        # this might kill the 'pure functionality' that gives JAX special powers

        adj_matrix.at[edge_0, edge_1].set(1)
        adj_matrix.at[edge_1, edge_0].set(1)

    if normalize:
        # make the diagonal node degree matrix
        d_matrix = jnp.diag(adj_matrix.sum(axis=-1))

        # invert it
        d_inv = jnp.linalg.inv(d_matrix)
        # take square root (Kipf and Welling 2017)
        #d_inv_sqrt = jnp.sqrt(d_inv)
        
        adj_matrix = jnp.dot(d_inv, adj_matrix)

    return adj_matrix
