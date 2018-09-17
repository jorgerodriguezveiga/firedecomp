"""Module with ResourcePeriod class definitions"""

# Package module
from firedecomp.classes import resources, wildfire


# ResourcePeriod --------------------------------------------------------------
class ResourcePeriod(object):
    """ResourcePeriod class."""

    def __init__(self, resource, period, resources_efficiency=1,
                 start=None, travel=None, rest=None, end_rest=None, end=None,
                 use=None, work=None):
        """Initialization of the class.

        Args:
            resource (:obj:`firedecomp.resources.Resource`): resource.
            period (:obj:`firedecomp.wildfire.Period`): period.
            resources_efficiency (:obj:`float`, opt): rosource efficiency
                (between 0 and 1). Where 1 is full performance and 0 is null
                performance. Defaults to 1.
            start (:obj:`bool`): ``True`` if the resource is starting to be
                used in this period. Defaults to ``None``.
            travel (:obj:`bool`): ``True`` if resource is traveling in this
                period. Defaults to ``None``.
            rest (:obj:`bool`): ``True`` if the resource is resting in this
                period. Defaults to ``None``.
            end_rest (:obj:`bool`): ``True`` if the resource is ending its
                break in this period. Defaults to ``None``.
            end (:obj:`bool`): ``True`` if the resource is ending its work in
                this period. Defaults to ``None``.
            use (:obj:`bool`): ``True`` if the resource has a task assigned in
                this period. Defaults to ``None``.
            work (:obj:`bool`): ``True`` if the resource works at fighting the
                wildfire in this period. Defaults to ``None``.

        TODO: Add example
        """
        self.resource = resource
        self.period = period
        self.resources_efficiency = resources_efficiency
        self.start = start
        self.travel = travel
        self.rest = rest
        self.end_rest = end_rest
        self.end = end
        self.use = use
        self.work = work

    def get_index(self):
        """Return index."""
        return self.resource.get_index(), self.period.get_index()

    def __repr__(self):
        index = self.get_index()
        return "<ResourcePeriod({}, {})>".format(
            index[0].__repr__(), index[1].__repr__())
# --------------------------------------------------------------------------- #


# ResourcesWildfire -----------------------------------------------------------
class ResourcesWildfire(object):
    """ResourcePeriod class."""

    def __init__(self, resources_wildfire):
        """Initialization of the class.

        Args:
            resources_wildfire (:obj:`list`): list of resources and wildfire
                objects (:obj:`ResourcePeriod`).

        TODO: Add example.
        """
        self.__index__ = self.__check_names__(resources_wildfire)
        self.resources_wildfire = resources_wildfire

    @staticmethod
    def __check_names__(resources_wildfire):
        """Check resource names."""
        indices = {n.get_index(): i for i, n in enumerate(resources_wildfire)}
        if len(indices) == len(resources_wildfire):
            return indices
        else:
            raise ValueError("ResourcePeriod name is repeated.")

    def get_names(self):
        return self.__index__.keys()

    def get_resource_period(self, i, p):
        """Get group by name.

        Args:
            i (:obj:`str` or `int`): resource name.
            p (:obj:`str` or `int`): period name.

        Return:
            :obj:`ResourcePeriod`: resource and period.
        """
        if (i, p) in self.get_index():
            return self.resources_wildfire[self.__index__[(i, p)]]
        else:
            raise ValueError(
                "Unknown group and period name: '({}, {})'".format(i, p))

    def get_resources(self):
        """Return resources."""
        return resources.Resources(
            list(set([rw.resource for rw in self.resources_wildfire])))

    def get_wildfire(self):
        """Return resources."""
        return wildfire.Wildfire(
            list(set([rw.wildfire for rw in self.resources_wildfire])))

    def __iter__(self):
        return (rw for rw in self.resources_wildfire)

    def __getitem__(self, key):
        return self.get_group_period(*key)

    def __repr__(self):
        return self.resources_wildfire.__repr__()
# --------------------------------------------------------------------------- #
