"""Module with GroupsWildfire definitions."""

# Python packages
import numpy as np


# GroupPeriod -----------------------------------------------------------------
class GroupPeriod(object):
    """GroupPeriod class."""

    def __init__(self, group, period, min_res_groups=0,
                 max_res_groups=np.infty, number_resources=None):
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
            number_resources (:obj:`int`): number of resources of the group.
                Defaults to ``None``.

        TODO: Add example.
        """
        self.group = group
        self.period = period
        self.min_res_groups = min_res_groups
        self.max_res_groups = max_res_groups
        self.number_resources = number_resources

    def get_index(self):
        """Return index."""
        return self.group.get_index(), self.period.get_index()

    def __repr__(self):
        index = self.get_index()
        return "<GroupPeriod({}, {})>".format(
            index[0].__repr__(), index[1].__repr__())
# --------------------------------------------------------------------------- #


# GroupsWildfire --------------------------------------------------------------
class GroupsWildfire(object):
    """GroupsWildfire class."""

    def __init__(self, groups_wildfire):
        """Initialization of the class.

        Args:
            groups_wildfire (:obj:`list`): list of groups and wildfire
                (:obj:`GroupPeriod`).

        TODO: Add example.
        """
        self.__index__ = self.__check_names__(groups_wildfire)
        self.groups_wildfire = groups_wildfire

    @staticmethod
    def __check_names__(groups_wildfire):
        """Check group and period names."""
        indices = {n.get_index(): i for i, n in enumerate(groups_wildfire)}
        if len(indices) == len(groups_wildfire):
            return indices
        else:
            raise ValueError("Group and period name is repeated.")

    def get_names(self):
        return self.__index__.keys()

    def get_group_period(self, g, p):
        """Get group and period object.

        Args:
            g (:obj:`str` or `int`): group name.
            p (:obj:`str` or `int`): period name.

        Return:
            :obj:`GroupPeriod`: group and period.
        """
        if (g, p) in self.get_names():
            return self.groups_wildfire[self.__index__[(g, p)]]
        else:
            raise ValueError(
                "Unknown group and period name: '({}, {})'".format(g, p))

    def __iter__(self):
        return (gw for gw in self.groups_wildfire)

    def __getitem__(self, key):
        return self.get_group_period(*key)

    def __repr__(self):
        return self.groups_wildfire.__repr__()
# --------------------------------------------------------------------------- #
