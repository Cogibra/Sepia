from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple

from sepia.graph.data import parse_edges_to_adjacency, GraphBatchTuple

GraphParametersWB = namedtuple("GraphParametersWB", ("weights", "biases"))

def graph_conv(batch: GraphBatchTuple, parameters: GraphParametersWB) -> GraphBatchTuple:
    """
    GraphBatchTuple includes (node_features, edges, edge_features, batch_indices)
    """
    adj = adjacency_matrix = parse_edges_to_adjacency(batch.edges, normalize="True")

    connected_nodes = jnp.dot(adj, batch.node_features)  

    nodes_out = jnp.dot(connected_nodes, parameters.weights)

    if "biases" in dir(parameters):
        new_node_features = nodes_out + parameters.biases
    else:
        new_node_features = nodes_out

    return new_node_features
    

def get_graph_conv_auto_loss(batch: GraphBatchTuple, parameters: GraphParametersWB) -> jnp.float32:

    target = 1.0 * batch.node_features

    node_features = batch.node_features
    edges = batch.edges
    edge_features = batch.edge_features
    batch_indices = batch.batch_indices

    for ii in range(3):
        my_batch = GraphBatchTuple(node_features, edges, edge_features, batch_indices) 
        node_features = graph_conv(my_batch, parameters)

    loss = jnp.mean((target - node_features)**2)

    return loss

grad_graph_conv_auto_loss = grad(get_graph_conv_auto_loss, argnums=(1), allow_int=True)

