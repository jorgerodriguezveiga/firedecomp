"""Module to define input class."""


# Input --------------------------------------------------------------------
class Input(object):
    """Resource class."""

    def __init__(self, resources, wildfire, law):
        """Initialization of the class.

        Args:
            resources (:obj:`Resources`): resources.
            wildfire (:obj:`Wildfire`): wildfire.
            law (:obj:`Law`): law.

        Example:
            >>> Air1 = Resource("Air1", group="Air", fix_cost=1000,
            >>>                 variable_cost=100, performance=100)
        """
        self.resources = resources
        self.wildfire = wildfire
        self.law = law
# --------------------------------------------------------------------------- #
