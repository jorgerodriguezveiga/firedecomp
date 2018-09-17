"""Module with Resources classes definitions."""


# Resource --------------------------------------------------------------------
class Resource(object):
    """Resource class."""

    def __init__(self, name, working_this_wildfire=False,
                 working_other_wildfire=False, arrival=0, work=0, rest=0,
                 total_work=0, performance=0, fix_cost=0, variable_cost=0,
                 time_between_rests=10, max_work_time=120,
                 necessary_rest_time=40, max_work_daily=480,
                 value=None
                 ):
        """Initialization of the class.

        Args:
            name (:obj:`str`): resource name.
            working_this_wildfire (:obj:`bool`): if True the resource is
                working in this wildfire. Defaults to ``False``.
            working_other_wildfire (:obj:`bool`): if True the resource is
                working in other wildfire. Defaults to ``False``.
            arrival (:obj:`int`): fly time to arrive to the wildfire (min).
            work (:obj:`int`): current work time (min).
            rest (:obj:`int`): current rest time (min).
            total_work (:obj:`int`): current use time (min).
            performance (:obj:`float`): performance (km/hour).
            fix_cost (:obj:`float`): fix cost (euro).
            variable_cost (:obj:`float`): variable cost (euro/hour).
            time_between_rests (:obj:`int`): time between rest point and
                wildfire (min).
            max_work_time (:obj:`int`): maximum time working without breaks
                (min). Defaults to ``120``.
            necessary_rest_time (:obj:`int`): necessary time of rest (min).
                Defaults to ``40``.
            max_work_daily (:obj:`int`): maximum daily time working
                (including rests, ...) (min). Defaults to ``480``.
            value (:obj:`bool`): ``True`` if resource is used in the wildfire.
                Defaults to ``None``.

        Example:
            >>> from firedecomp.classes.resources import Resource
            >>> Air1 = Resource(
            >>>     "Air1", fix_cost=1000, variable_cost=100, performance=100)

        TODO: Check input: integrality, positivity, ...
        """
        self.name = name
        self.working_this_wildfire = working_this_wildfire
        self.working_other_wildfire = working_other_wildfire
        self.arrival = arrival
        self.work = work
        self.rest = rest
        self.total_work = total_work
        self.performance = performance
        self.fix_cost = fix_cost
        self.variable_cost = variable_cost
        self.time_between_rests = time_between_rests
        self.max_work_time = max_work_time
        self.necessary_rest_time = necessary_rest_time
        self.max_work_daily = max_work_daily
        self.value = value

    def get_index(self):
        """Return index."""
        return self.name

    def __repr__(self):
        return "<Resource({})>".format(self.name.__repr__())
# --------------------------------------------------------------------------- #


# Resources -------------------------------------------------------------------
class Resources(object):
    """Resources class."""

    def __init__(self, resources):
        """Initialization of the class.

        Args:
            resources (:obj:`list`): list of resources (:obj:`Resource`).

        Example:
            >>> from firedecomp.classes.resources import Resources, Resource
            >>> Air1 = Resource("Air1", fix_cost=1000,
            >>>                 variable_cost=100, performance=100)
            >>> Air2 = Resource("Air2", fix_cost=1500,
            >>>                 variable_cost=80, performance=100)
            >>> Bri1 = Resource("Bri1", fix_cost=100,
            >>>                 variable_cost=50, performance=100,
            >>>                 necessary_rest_time=0, max_work_time=480)
            >>> resources = Resources(resources=[Air1, Air2, Bri1])
        """
        self.__index__ = self.__check_names__(resources)
        self.resources = resources

    @staticmethod
    def __check_names__(resources):
        """Check resource names."""
        indices = {n.get_index(): i for i, n in enumerate(resources)}
        if len(resources) == len(indices):
            return indices
        else:
            raise ValueError("Resource name is repeated.")

    def get_names(self):
        return self.__index__.keys()

    def get_resource(self, i):
        """Get resource by name.

        Args:
            i (:obj:`str`): resource name.

        Return:
            :obj:`Resource`: resource.
        """
        if i in self.get_names():
            return self.resources[self.__index__[i]]
        else:
            raise ValueError("Unknown resource name: '{}'".format(i))

    def __iter__(self):
        return (r for r in self.resources)

    def __getitem__(self, key):
        return self.get_resource(key)

    def __repr__(self):
        return self.resources.__repr__()
# --------------------------------------------------------------------------- #
