import argparse
import os

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from collections import namedtuple


def sgd(gradient: jnp.array, info: type(None)=None) -> tuple:
    """
    optimizer step function. 

    takes a gradient parameter array and info, returns update array
    
    This update function is basically an identity function, because SGD has no memory.

    """

    return (gradient, info)

def adam(gradient: jnp.array, info: tuple=None) -> tuple:
    """
    optimizer step function. 

    takes a gradient parameter array and info, returns update array
    
    This update function is basically an identity function, because SGD has no memory.

    info is a tuple consisting of 
        (beta_0, beta_1, moment, moment_2
        beta_0: exponential averaging variable for first moment
        beta_1: exponential averaging variable for second moment
        moment: first moment (exponential average of gradient)
        moment_2: second moment (exponential average of gradient^2

    """

    if info is not None:
        moment = info[0] * info[2] + (1-info[0]) * gradient 
        moment_2 = info[1] * info[3] + (1-info[1]) * gradient**2
        beta_0, beta_1 = info[0], info[1] 
    else:
        beta_0, beta_1 = 1e-3, 1e-4
        moment = gradient
        moment_2 = gradient**2

    info = (beta_0, beta_1, moment, moment_2)

    return (gradient, info)


def step(parameters: namedtuple, gradients: namedtuple, \
        lr: float=1e-4, update: type(sgd)=sgd, info: tuple=None):
    new_parameters = []

    new_params = {}
    for ii, (param, grad) in enumerate(zip(parameters, gradients)):
        
        # indirect workaround for testing whether the param is a namedtuple
        if dir(param)[35] == "_fields":
            new_params[parameters._fields[ii]] = step(param, grad, lr, update, info)[0]
        else:
            param_update, info = update(grad, info)
            new_params[parameters._fields[ii]] = param - lr * param_update

    new_parameters = type(parameters)(**new_params)

    return new_parameters, info



