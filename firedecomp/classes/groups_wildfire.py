"""Module with GroupsWildfire definitions."""

# Python packages
import numpy as np

# Package modules
from firedecomp.classes import general as gc


# GroupPeriod -----------------------------------------------------------------
class GroupPeriod(gc.Element):
    """GroupPeriod class."""

    def __init__(self, group, period, min_res_groups=0,
                 max_res_groups=np.infty, num_left_resources=None):
        """Initialization of the class.

        Args:
            group (:obj:`firedecomp.groups.Group`): group.
            period (:obj:`firedecomp.wildfire.Period`): wildfire.
            min_res_groups (:obj:`int`, opt): minimum number of resources of
                each
                group working on the wildfire in the period. Defaults to
                ``None``.
            max_res_groups (:obj:`dict`): maximum number of resources of each
                group working on the wildfire in the period. Defaults to
                ``None``.
            num_left_resources (:obj:`int`): number of left resources.
                Defaults to ``None``.

        TODO: Add example.
        """
        super(GroupPeriod, self).__init__(
            name=(group.get_index(), period.get_index()))
        self.group = group
        self.period = period
        self.min_res_groups = min_res_groups
        self.max_res_groups = max_res_groups
        self.num_left_resources = num_left_resources
# --------------------------------------------------------------------------- #


# GroupsWildfire --------------------------------------------------------------
class GroupsWildfire(gc.Set):
    """GroupsWildfire class."""

    def __init__(self, groups_wildfire):
        """Initialization of the class.

        Args:
            groups_wildfire (:obj:`list`): list of groups and wildfire
                (:obj:`GroupPeriod`).

        TODO: Add example.
        """
        super(GroupsWildfire, self).__init__(elements=groups_wildfire)
# --------------------------------------------------------------------------- #
