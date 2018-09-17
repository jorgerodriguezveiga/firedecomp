"""Wildfire module."""


# Period --------------------------------------------------------------------
class Period(object):
    """Period problem input class."""

    def __init__(self, name, perimeter, cost, contained=None):
        """Initialization of the class.

        Args:
            name (:obj:`int`): period name.
            perimeter (:obj:`float`): perimeter of the wildfire (km).
            cost (:obj:`float`): cost of the wildfire related to affected area
                costs, reforestation, urbane damage, ... (euro).
            contained (:obj:`bool`): ``True`` if wildfire is contained in this
                period.

        Example:
            >>> from firedecomp.classes.wildfire import Period
            >>> p1 = Period(1, perimeter=150, cost=100)

        TODO: Check if name is int and perimeter and cost floats or int.
        """
        self.name = name
        self.perimeter = perimeter
        self.increment_perimeter = None
        self.cost = cost
        self.increment_cost = None
        self.contained = contained

    def get_index(self):
        """Return index."""
        return self.name

    def __repr__(self):
        return "<Period({})>".format(str(self.name.__repr__()))
# --------------------------------------------------------------------------- #


# Wildfire --------------------------------------------------------------------
class Wildfire(object):
    """Wildfire class."""

    def __init__(self, periods, time_per_period=10):
        """Initialization of the class.

        Args:
            periods (:obj:`list`): list of periods with wildfire information
                (:obj:`Period`).
            time_per_period (:obj:`int`): time comprising a period (min).
                Defaults to ``10``.

        Example:
            >>> from firedecomp.classes.wildfire import Period, Wildfire
            >>> p1 = Period(1, increment_perimeter=150, increment_cost=100)
            >>> p2 = Period(2, increment_perimeter=200, increment_cost=150)
            >>> p3 = Period(3, increment_perimeter=150, increment_cost=150)
            >>> wildfire = Wildfire([p1, p2, p3])
        """
        self.__index__ = self.__check_names__(periods)
        self.periods = periods
        self.time_per_period = time_per_period

        # Compute perimeter and cost increments
        self.compute_increments()

    @staticmethod
    def __check_names__(periods):
        """Check periods names."""
        indices = {p.get_index(): i for i, p in enumerate(periods)}
        if len(indices) == len(periods):
            return indices
        else:
            raise ValueError("Period name is repeated.")

    def get_names(self):
        return list(self.__index__.keys())

    def get_period(self, p):
        """Get period by name.

        Args:
            p (:obj:`str` or `int`): period name.

        Return:
            :obj:`Period`: period.
        """
        if p in self.get_names():
            return self.periods[self.__index__[p]]
        else:
            raise ValueError("Unknown period name: '{}'".format(p))

    def get_periods(self, p_min=None, p_max=None):
        """Get periods between p_min and p_max.

        Args:
            p_min (:obj:`int`, opt): start period. If None, first one is taken.
                Defaults to ``None``.
            p_max (:obj:`int`, opt): end period. If None, last one is taken.
                Defaults to ``None``.

        Return:
            :obj:`list`: list of periods.
        """
        periods_names = self.get_names()

        # min period
        min_period = min(periods_names)
        if p_min is None:
            p_min = min_period
        elif p_min <= min_period:
            p_min = min_period

        # max period
        max_period = max(periods_names)
        if p_max is None:
            p_max = max_period
        elif p_max <= max_period:
            p_max = max_period

        return [self.get_period(p) for p in range(p_min, p_max+1)]

    def compute_increments(self):
        """Compute perimeter and cost increment."""
        names_period = self.get_names()
        first_period = min(names_period)

        p = self.get_period(first_period)

        p.increment_cost = p.cost
        p.increment_perimeter = p.perimeter

        cost_prev = p.cost
        perimeter_prev = p.perimeter

        for p in self.get_periods(p_min=first_period+1):

            p.increment_cost = p.cost - cost_prev
            p.increment_perimeter = p.perimeter - perimeter_prev

            cost_prev = p.cost
            perimeter_prev = p.perimeter

    def size(self):
        return len(self.periods)

    def __iter__(self):
        return (p for p in self.periods)

    def __getitem__(self, key):
        return self.get_period(key)

    def __repr__(self):
        return self.periods.__repr__()
# --------------------------------------------------------------------------- #
