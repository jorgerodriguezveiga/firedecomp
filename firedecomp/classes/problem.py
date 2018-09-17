"""Module to define input class."""


# Problem ---------------------------------------------------------------------
class Problem(object):
    """Problem class."""

    def __init__(
            self, resources, wildfire, groups,
            groups_wildfire=None, resources_wildfire=None):
        """Problem class.

        Args:
            resources
            wildfire
            groups_wildfire (:obj:`GroupsWildfire`): groups wildfire.
            resources_wildfire (:obj:`ResourcesWildfire`): resources wildfire.

        TODO: Add example.
        TODO: Create class to convert to period units.
        """
        self.resources = resources
        self.wildfire = wildfire
        self.groups = groups
        self.groups_wildfire = groups_wildfire
        self.resources_wildfire = resources_wildfire

    def __repr__(self):
        return "PERIODS: {}\n" \
               "RESOURCES: {}".format(self.wildfire.size(),
                                      self.resources.__repr__())

    def get_periods_problem(self):
        """Transform problem units.

        TODO: Transform problem units from minutes to periods.
        """
        return self
# --------------------------------------------------------------------------- #
