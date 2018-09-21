"""Module with group definition."""

# Package modules
from firedecomp.classes import general_classes as gc


# Group -----------------------------------------------------------------------
class Group(gc.Element):
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
        super(Group, self).__init__(name=name)
        self.resources = resources

    def size(self):
        """Return the number of members of the group."""
        return len(list(self.resources))
# --------------------------------------------------------------------------- #


# Groups ----------------------------------------------------------------------
class Groups(gc.Set):
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
        super(Groups, self).__init__(elements=groups)
# --------------------------------------------------------------------------- #
