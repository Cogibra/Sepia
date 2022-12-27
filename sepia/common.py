import argparse
import os

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple

def query_kwargs(key: str, default: any, **kwargs):

    if key in kwargs.keys():
        return kwargs[key]
    else:
        return default

def optimizer_step(parameters: namedtuple, gradients: namedtuple, lr: float=1e-4):
    # SGD 

    new_parameters = []

    new_params = {}
    for ii, (param, grad) in enumerate(zip(parameters, gradients)):
        
        # indirect workaround for testing whether the param is a namedtuple
        if dir(param)[35] == "_fields":

            new_params[parameters._fields[ii]] = optimizer_step(param, grad, lr)
        else:

            new_params[parameters._fields[ii]] = param - lr * grad

    try:
        new_parameters = type(parameters)(**new_params)
    except:
        import pdb; pdb.set_trace()

    return new_parameters



