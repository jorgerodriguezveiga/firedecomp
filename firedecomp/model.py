"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy

# Package modules
from firedecomp.classes import solution


class InputModel(object):
    def __init__(self, problem_data):
        if problem_data.period_unit is False:
            raise ValueError("Time unit of the problem is not a period.")

        self.data = problem_data


def model(model_data):
    """Wildfire suppression model.
    model_data (:obj:`firedecomp.model.InputModel`): problem data.
    """