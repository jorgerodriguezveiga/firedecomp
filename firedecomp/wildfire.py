"""Wildfire module."""


# Wildfire --------------------------------------------------------------------
class Wildfire(object):
    """Wildfire class."""

    def __init__(self, periods):
        """Initialization of the class.

        Args:
            periods (:obj:`list`): list of periods with wildfire information
                (:obj:`Period`).

        Example:
            >>> per1 = Period("Air1", group="Air", fix_cost=1000,
            >>>                 variable_cost=100, performance=100)
            >>> per2 = Period("Air2", group="Air", fix_cost=1500,
            >>>                 variable_cost=80, performance=100)
            >>> per3 = Period("Bri1", group="Bri", fix_cost=100,
            >>>                 variable_cost=50, performance=100)
            >>> resources = Resources([Air1, Air2, Bri1])
        TODO: Check input: different names.
        """
        # Names
        self.__names__ = self.__check_names__(resources)
        self.__names_idx__ = {n: i for i, n in enumerate(self.__names__)}

        # Groups
        self.__groups__ = [r.group for r in self.resources]
        groups_idx = {g: [] for g in self.__groups__}
        for i, r in enumerate(resources):
            groups_idx[r.group].append(i)
        self.__groups_idx__ = groups_idx

        # Resources
        self.resources = resources

    @staticmethod
    def __check_names__(resources):
        """Check resource names."""
        names_list = [getattr(r, 'name') for r in resources]
        names_set = list(set(names_list))
        if len(names_list) == len(names_set):
            return names_set
        else:
            raise ValueError("Resource name is repeated.")

    def get_names(self):
        return self.__names__

    def get_groups(self):
        """List of groups."""
        return self.__groups__

    def get_resource(self, name):
        """Get resource by name.

        Args:
            name (:obj:`str`): resource name.

        Return:
            :obj:`Resource`: resource.
        """
        if name in self.get_names():
            return self.resources[self.__names_idx__[name]]
        else:
            return None

    def get_resource_info(self, name, info="name"):
        """Get resource information by name.

        Args:
            name (:obj:`str`): resource name.
            info (:obj:`str`): attribute information. Defaults to "name".

        Return:
            :obj:`Resource`: resource.
        """
        resource = self.get_resource(name)
        if resource is not None:
            return getattr(resource, info)
        else:
            return None

    def get_resources(self, tag, by="name"):
        """Get a list of resources.

        Args:
            tag (:obj:`str`): tag name.
            by (:obj:`str`): options "name" or "group". Defaults to ``"name"``.

        Return:
            :obj:`Resources`of res
        """
        if by == "name":
            return self.get_resource(name=tag)
        elif by == "group":
            return Resources([self.resources[i]
                              for i in self.__groups_idx__[tag]])
        else:
            raise ValueError("Wrong by value: {}".format(by))

    def to_list(self):
        return self.resources
# --------------------------------------------------------------------------- #


# Period --------------------------------------------------------------------
class Period(object):
    """Period problem input class."""

    def __init__(self, name, group=None, fix_cost=0, variable_cost=0,
                 performance=0, arrival=0, rest=0, work=0, total_work=0,
                 use=False):
        """Initialization of the class.

        Args:
            name (:obj:`str`): resource name.
            group (:obj:`str` or `int`): group name.
            fix_cost (:obj:`float`): fix cost (euro).
            variable_cost (:obj:`float`): variable cost (euro/hour).
            performance (:obj:`float`): performance (km/hour).
            arrival (:obj:`float`): hours of fly to arrive to the wildfire.
            rest (:obj:`float`): current rest time (hours).
            work (:obj:`float`): current work time (hours).
            total_work (:obj:`float`): current use time (hours).
            use (:obj:`bool`): if True indicate it is being used.

        Example:
            >>> Air1 = Resource("Air1", group="Air", fix_cost=1000,
            >>>                 variable_cost=100, performance=100)
        """
        self.name = name
        self.group = group
        self.fix_cost = fix_cost
        self.variable_cost = variable_cost
        self.performance = performance
        self.arrival = arrival
        self.rest = rest
        self.work = work
        self.total_work = total_work
        self.use = use
# --------------------------------------------------------------------------- #
