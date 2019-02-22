"""Wildfire module."""

# Package modules
from firedecomp.classes import general as gc
from firedecomp.classes import resources_wildfire as rw


# Period ----------------------------------------------------------------------
class Period(gc.Element):
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
        super(Period, self).__init__(name=name)
        self.perimeter = perimeter
        self.increment_perimeter = None
        self.cost = cost
        self.increment_cost = None
        self.contained = contained
        self.__resource_period__ = rw.ResourcesWildfire([])

    def get_increment_perimeter(self):
        return self.increment_perimeter * (not self.contained)
# --------------------------------------------------------------------------- #


# Wildfire --------------------------------------------------------------------
class Wildfire(gc.Set):
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
        super(Wildfire, self).__init__(elements=periods)
        self.time_per_period = time_per_period

        # Compute perimeter and cost increments
        self.compute_increments()

    def get_min(self):
        periods_names = self.get_names()

        if len(periods_names) == 0:
            None
        elif len(periods_names) == 1:
            min_period = periods_names[0]
        else:
            min_period = int(min(periods_names))
        return self.get_element(min_period)

    def get_max(self):
        periods_names = self.get_names()

        if len(periods_names) == 0:
            None
        elif len(periods_names) == 1:
            max_period = periods_names[0]
        else:
            max_period = int(max(periods_names))
        return self.get_element(max_period)

    def get_names(self, p_min=None, p_max=None):
        if p_min is None and p_max is None:
            return list(self.__index__.keys())
        else:
            return [e.name for e in self.get_elements(p_min=p_min, p_max=p_max)]

    def get_elements(self, p_min=None, p_max=None):
        """Get elements between p_min and p_max.

        Args:
            p_min (:obj:`int`, opt): start period. If None, first one is taken.
                Defaults to ``None``.
            p_max (:obj:`int`, opt): end period. If None, last one is taken.
                Defaults to ``None``.

        Return:
            :obj:`list`: list of periods.
        """
        # min period
        min_period = self.get_min().name
        if p_min is None:
            p_min = min_period
        elif p_min <= min_period:
            p_min = min_period

        # max period
        max_period = self.get_max().name
        if p_max is None:
            p_max = max_period
        elif p_max >= max_period:
            p_max = max_period

        return [self.get_element(p) for p in range(int(p_min), int(p_max+1))]

    def compute_increments(self):
        """Compute perimeter and cost increment."""
        p = self.get_min()
        first_period = p.name

        p.increment_cost = p.cost
        p.increment_perimeter = p.perimeter

        cost_prev = p.cost
        perimeter_prev = p.perimeter

        for p in self.get_elements(p_min=first_period+1):

            p.increment_cost = p.cost - cost_prev
            p.increment_perimeter = p.perimeter - perimeter_prev

            cost_prev = p.cost
            perimeter_prev = p.perimeter

    def get_contention_perimeter(self):
        """Compute total perimeter before the wildfire is contained."""
        return sum([p.get_increment_perimeter() for p in self])
# --------------------------------------------------------------------------- #
