"""Wildfire module."""

# Package modules
from firedecomp.classes import general_classes as gc


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

        return [self.get_element(p) for p in range(p_min, p_max+1)]

    def compute_increments(self):
        """Compute perimeter and cost increment."""
        names_period = self.get_names()
        first_period = min(names_period)

        p = self.get_element(first_period)

        p.increment_cost = p.cost
        p.increment_perimeter = p.perimeter

        cost_prev = p.cost
        perimeter_prev = p.perimeter

        for p in self.get_elements(p_min=first_period+1):

            p.increment_cost = p.cost - cost_prev
            p.increment_perimeter = p.perimeter - perimeter_prev

            cost_prev = p.cost
            perimeter_prev = p.perimeter
# --------------------------------------------------------------------------- #
