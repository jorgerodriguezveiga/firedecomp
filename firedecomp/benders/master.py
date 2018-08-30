"""Module with Master problem class."""


# InputMaster -----------------------------------------------------------------
class Master(object):
    """Master problem input class."""

    def __init__(self, resources, periods, legislation):
        """Initialization of the class.

        Args:
            model (:obj:`gurobipy.Model` or `list`): gurobipy model or list of
                pyomo models.
            variables (:obj:`dict`): dictionary with gurobipy variables where
                keys of the dict are considered variable names.
        """
        self.model = model
        self.variables = Variables(variables)
# --------------------------------------------------------------------------- #

# Master ----------------------------------------------------------------------
class Master(object):
    """Master problem class."""

    def __init__(self, model, variables=None):
        """Initialization of the class.

        Args:
            model (:obj:`gurobipy.Model` or `list`): gurobipy model or list of
                pyomo models.
            variables (:obj:`dict`): dictionary with gurobipy variables where
                keys of the dict are considered variable names.
        """
        self.model = model
        self.variables = Variables(variables)