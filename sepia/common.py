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

