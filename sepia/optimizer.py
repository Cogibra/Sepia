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
    
    adam updates calculate the first and second moments of the gradient
    the moments and exponential averaging parameters are contained in info. 

    info is a tuple consisting of 
        (beta_0, beta_1, moment, moment_2
        beta_0: exponential averaging variable for first moment
        beta_1: exponential averaging variable for second moment
        moment: first moment (exponential average of gradient)
        moment_2: second moment (exponential average of gradient^2
    """

    beta_0, beta_1 = 1e-3, 1e-4
    if info is not None:
        if gradient.shape[0] != info[1].shape[0]:
            assert False, f"gradient shape {gradient.shape} does not match moment shape {info[2].shape}"
        moment = beta_0 * info[1] + (1-beta_0) * gradient 
        moment_2 = beta_1 * info[2] + (1-beta_1) * gradient**2
    else:
        moment = gradient
        moment_2 = gradient**2

    info = jnp.append(moment[None,...], moment_2[None,...], axis=0)

    return (gradient, info)


def step(parameters: namedtuple, gradients: namedtuple, \
        lr: float=1e-4, update: type(sgd)=sgd, info: tuple=None):
    new_parameters = []

    new_params = {}
    info_params = {}
    if info is None:
        info = [None] * len(parameters)
    for ii, (param, grad, my_info) in enumerate(zip(parameters, gradients, info)):
        
        # indirect workaround for testing whether the param is a namedtuple
        if "_field" in dir(param)[35]:
            new_params[parameters._fields[ii]], \
                    info_params[parameters._fields[ii]] = \
                    step(param, grad, lr, update, my_info)
        else:
            param_update, new_info = update(grad, my_info)
            new_params[parameters._fields[ii]] = param - lr * param_update
            info_params[parameters._fields[ii]] = new_info

    new_parameters = type(parameters)(**new_params)
    new_info = type(parameters)(**info_params)

    return new_parameters, new_info
