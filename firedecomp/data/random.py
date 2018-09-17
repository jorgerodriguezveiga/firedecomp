"""Module to generate random objects."""

# Python package
import numpy as np


# random_num ------------------------------------------------------------------
def random_num(min_val, max_val, seed=None, zero=1):
    """Generate random number between min and max."""
    if seed is not None:
        np.random.seed(seed)
    return round(
        (min_val + np.random.random()*(max_val-min_val))/10**zero
    )*10**zero
# --------------------------------------------------------------------------- #
