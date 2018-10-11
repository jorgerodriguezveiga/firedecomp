"""Module with model class."""


# Solution --------------------------------------------------------------------
class Solution(object):
    """Solution class."""

    def __init__(self, model, variables=None):
        """Solution class.

        Args:
            model (:obj:`gurobipy.Model` or `list`): gurobipy model or list of
                pyomo models.
            variables (:obj:`dict`): dictionary with gurobipy variables where
                keys of the dict are considered variable names.
        """
        self.model = model
        self.variables = Variables(variables)

    def set_model(self, model):
        """Set the gurobipy model.

        Args:
            model (:obj:`gurobipy.Model`): gurobipy model.
        """
        self.model = model

    def set_variables(self, variables):
        """Set the variables of the gurobipy model.

        Args:
            variables (:obj:`dict`): dictionary with gurobipy variables where
                keys of the dict are considered variable names.
        """
        self.variables.set_variables(variables)

    def get_model(self):
        """Return the gurobipy model.

        Return:
            :obj:`gurobipy.Model`: gurobipy model.
        """
        return self.model

    def get_variables(self):
        """Return the variables of the gurobipy model.

        Return:
            :obj:`pynanos.model.Variables`: variables object.
        """
        return self.variables
# --------------------------------------------------------------------------- #


# Variables -------------------------------------------------------------------
class Variables(object):
    """Variables class."""

    def __init__(self, variables=None):
        """Variables class.

        Args:
            variables (:obj:`dict`): dictionary with gurobipy variables where
                keys of the dict are considered variable names. If None no
                variables are considered. Defaults to None.
        """
        self.__names__ = []
        self.set_variables(variables)

    def set_variables(self, variables):
        """Set the variables of the gurobipy model.

        Args:
            variables (:obj:`dict`): dictionary with gurobipy variables where
                keys of the dict are considered variable names.
        """
        if variables is not None:
            for k in variables:
                self.set_variable(k, variables[k])

    def set_variable(self, name, var):
        """Set a variable of the gurobipy model.

        Args:
            name (:obj:`str`): dictionary with gurobipy variables where
                keys of the dict are considered variable names.
            var (:obj:`gurobipy.tupledict` or `gurobipy.Var`): gurobipy
                variable.
        """
        setattr(self, name, var)
        self.__names__.append(name)

    def get_names(self):
        """Return list of variable names.

        Return:
            :obj:`list`: list of variable names.
        """
        return self.__names__

    def get_variable(self, name):
        """Return a variable.

        Args:
            name (:obj:`str`): variable name.

        Return:
            :obj:`gurobipy.tupledict` or `gurobipy.Var`: gurobipy variable.
        """
        return getattr(self, name)

    def get_variables(self, names=None):
        """Return a list of variables.

        Args:
            names (:obj:`list`): list of strings with variable names. If None
                all variables are returned. Defaults to None.

        Return:
            :obj:`list`: list of gurobipy variables.
        """
        if names is None:
            return [self.get_variable(v) for v in self.get_names()]
        else:
            return [self.get_variable(v) for v in names]
# --------------------------------------------------------------------------- #
