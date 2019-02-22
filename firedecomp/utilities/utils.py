
# Python Packages
import os
import yaml
import math
import logging as log


# get_file_directory ----------------------------------------------------------
def get_file_directory(file):
    return os.path.dirname(os.path.abspath(file))
# --------------------------------------------------------------------------- #


# join_paths ------------------------------------------------------------------
def join_paths(path, *paths):
    return os.path.abspath(os.path.join(path, *paths))
# --------------------------------------------------------------------------- #


# load_yaml -------------------------------------------------------------------
def load_yaml(path):
    return yaml.load(open(path).read())
# --------------------------------------------------------------------------- #


# num_between_floor_ceil ------------------------------------------------------
def num_between_floor_ceil(initial, result, tol=0.0001, error=''):
    """Check if a number is between the floor and ceil of an initial value.

    The condition to check is:
        floor(initial)-tol <= result <= ceil(initial)+tol

    Args:
        initial (:obj:`float`): initial value to compute its floor and ceil.
        result (:obj:`float`): value to check if it is between the initial
            value floor and ceil.
        tol (:obj:`float`): tolerance value. Defaults to 0.0001.
        error (:obj:`str`): error message if the condition is not True. Default
            ''.

    Return:
        :obj:`bool`: True if the condition is met. False otherwise.
    """
    floor = math.floor(initial) - tol
    ceil = math.ceil(initial) + tol
    if floor <= result <= ceil:
        return True
    else:
        log.error(error)
        log.error('{} <= {} <= {}'.format(floor, result, ceil))
        return False
# --------------------------------------------------------------------------- #


# is_integer ------------------------------------------------------------------
def is_integer(value, tol=0.0001, error=''):
    """Check if value is integer.

    Args:
        value (:obj:`float`): number to check integrality.
        tol (:obj:`float`): tolerance value. Defaults to 0.0001.
        error (:obj:`str`): error message if the condition is not True. Default
            ''.

    Return:
        :obj:`bool`: True if the condition is met. False otherwise.
    """
    round_val = round(value)
    if round_val - tol <= value <= round_val + tol:
        return True
    else:
        log.error(error)
        return False
# --------------------------------------------------------------------------- #
