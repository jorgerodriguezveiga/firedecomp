"""Module with group definition."""


# Group -----------------------------------------------------------------------
class Group(object):
    """Group class."""

    def __init__(self, name, resources):
        """Group of resources.

         Group of resources with similar characteristics.

        Args:
            name (:obj:`str` or `int`): group name.
            resources (:obj:`firedecomp.classes.resources.Resources`):
                resources group.

        Example:
            >>> import firedecomp
            >>> res = firedecomp.data.examples.resources_example(
            >>>     num_machines=5, num_aircraft=0, num_brigades=0)
            >>> group = firedecomp.classes.groups.Group(name=1, resources=res)
        """
        self.name = name
        self.resources = resources

    def get_index(self):
        """Return index."""
        return self.name

    def size(self):
        """Return the number of members of the group."""
        return len(list(self.resources))

    def __repr__(self):
        return "<Group({})>".format(self.name.__repr__())
# --------------------------------------------------------------------------- #


# Groups ----------------------------------------------------------------------
class Groups(object):
    """Groups class."""

    def __init__(self, groups):
        """Collection of groups.

        Args:
            groups (:obj:`list`): list of groups (:obj:`Group`).

        Example:
            >>> import firedecomp
            >>> brigades = firedecomp.data.examples.resources_example(
            >>>     num_brigades=5,
            >>>     num_aircraft=0,
            >>>     num_machines=0)
            >>> aircraft = firedecomp.data.examples.resources_example(
            >>>     num_brigades=0,
            >>>     num_aircraft=5,
            >>>     num_machines=0)
            >>> brigades_grp = firedecomp.classes.groups.Group(
            >>>     name="brigades", resources=brigades)
            >>> aircraft_grp = firedecomp.classes.groups.Group(
            >>>     name="aircraft", resources=aircraft)
            >>> groups = firedecomp.classes.groups.Groups(
            >>>     [brigades_grp, aircraft_grp])
        """
        self.__index__ = self.__check_names__(groups)
        self.groups = groups

    @staticmethod
    def __check_names__(groups):
        """Check resource names."""
        indices = {n.get_index(): i for i, n in enumerate(groups)}
        if len(groups) == len(indices):
            return indices
        else:
            raise ValueError("Group name is repeated.")

    def get_names(self):
        return self.__index__.keys()

    def get_group(self, g):
        """Get group by name.

        Args:
            g (:obj:`str` or `int`): group name.

        Return:
            :obj:`Group`: group.
        """
        if g in self.get_names():
            return self.groups[self.__index__[g]]
        else:
            raise ValueError("Unknown group name: '{}'".format(g))

    def __iter__(self):
        return (g for g in self.groups)

    def __getitem__(self, key):
        return self.get_group(key)

    def __repr__(self):
        return self.groups.__repr__()
# --------------------------------------------------------------------------- #
