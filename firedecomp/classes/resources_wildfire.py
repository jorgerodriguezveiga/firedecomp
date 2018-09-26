"""Module with ResourcePeriod class definitions"""

# Package modules
from firedecomp.classes import general as gc


# ResourcePeriod --------------------------------------------------------------
class ResourcePeriod(gc.Element):
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
        super(ResourcePeriod, self).__init__(
            name=(resource.get_index(), period.get_index()))
        self.resource = resource
        self.period = period
        self.resources_efficiency = resources_efficiency
        self.resource_performance = resources_efficiency*resource.performance
        self.start = start
        self.travel = travel
        self.rest = rest
        self.end_rest = end_rest
        self.end = end
        self.use = use
        self.work = work

    def update_units(self, time, period_unit, inplace=True):
        """Update resource attributes units.

        Args:
            period_unit (:obj:`bool`): if ``True`` period units. Defaults to
                ``True``.
            time (:obj:`int`): minutes corresponding a period.
            inplace (:obj:`bool`): if ``True``, perform operation in-place.
                Defaults to ``False``.
        """
        if inplace is True:
            resource_period = self
        else:
            resource_period = self.copy()

        if period_unit is True:
            prop_time_hour = 60/time
        else:
            prop_time_hour = time/60

        self.resource_performance = self.resource_performance/prop_time_hour

        if inplace is not True:
            return resource_period
# --------------------------------------------------------------------------- #


# ResourcesWildfire -----------------------------------------------------------
class ResourcesWildfire(gc.Set):
    """ResourcePeriod class."""

    def __init__(self, resources_wildfire):
        """Initialization of the class.

        Args:
            resources_wildfire (:obj:`list`): list of resources and wildfire
                objects (:obj:`ResourcePeriod`).

        TODO: Add example.
        """
        super(ResourcesWildfire, self).__init__(elements=resources_wildfire)

    def update_units(self, time, period_unit=True, inplace=False):
        """Update wildfire_resources attributes units.

        Args:
            time (:obj:`int`): minutes corresponding a period.
            period_unit (:obj:`bool`): if ``True`` period units. Defaults to
                ``True``.
            inplace (:obj:`bool`): if ``True``, perform operation in-place.
                Defaults to ``False``.
        """
        for e in self:
            e.update_units(time=time, period_unit=period_unit, inplace=inplace)
# --------------------------------------------------------------------------- #
