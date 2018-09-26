"""Module to define input class."""

# Package modules
from firedecomp.original import model
from firedecomp import config

# Problem ---------------------------------------------------------------------
class Problem(object):
    """Problem class."""

    def __init__(
            self, resources, wildfire, groups,
            groups_wildfire=None, resources_wildfire=None, period_unit=False):
        """Problem class.

        Args:
            resources
            wildfire
            groups_wildfire (:obj:`GroupsWildfire`): groups wildfire.
            resources_wildfire (:obj:`ResourcesWildfire`): resources wildfire.
            period_unit (:obj:`bool`): ``True`` if time is measured in periods.
                Defaults to ``False``.

        TODO: Add example.
        TODO: Create class to convert to period units.
        """
        self.resources = resources
        self.wildfire = wildfire
        self.groups = groups
        self.groups_wildfire = groups_wildfire
        self.resources_wildfire = resources_wildfire
        self.period_unit = period_unit
        self.solve_status = None

    def update_units(self, period_unit=True, inplace=True):
        """Update problem units.

        Args:
            period_unit (:obj:`bool`): if ``True`` period units. Defaults to
                ``True``.
            inplace (:obj:`bool`): if ``True`` modify the object in place (do
                not create a new object).
        """
        if inplace is True:
            problem = self
        else:
            problem = self.copy()

        if period_unit != self.period_unit:
            # Get time per period
            time = self.wildfire.time_per_period

            # Update resource attributes units
            problem.resources.update_units(
                time=time, period_unit=period_unit, inplace=True)
            problem.resources_wildfire.update_units(
                time=time, period_unit=period_unit, inplace=True)

            # Update units status
            problem.period_unit = period_unit

        if inplace is not True:
            return problem

    def status(self, info="status"):
        """Return resolution status.

        Args:
            info (:obj:`str`): status information. Options: ``'status'``,
                ``'status_code'`` or ``'description'``. Defaults to
                ``'status'``.
        """
        if info == 'status':
            return self.solve_status
        else:
            return config.gurobi.status_info[self.status][info]

    def solve(self, method='original', solver_options=None,
              min_res_penalty=1000000):
        """Solve mathematical model.

        Args:
            method (:obj:`str`): method to solve the mathematical problem.
                Options: ``'original'``. Defaults to ``'original'``.
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
            min_res_penalty (:obj:`float`): positive value that penalize the
                breach of the minimum number of resources in each period.
                Defaults to ``1000000``.
        """
        self.update_units()
        if method == 'original':
            m = model.InputModel(self, min_res_penalty=min_res_penalty)
            solution = m.solve(solver_options=solver_options)
            self.solve_status = solution.model.Status
            return solution
        else:
            raise ValueError(
                "Incorrect method '{}'. Options allowed: {}".format(
                    method, ["original"]
                ))

    def plot(self, info=['contantion', 'performance', 'resources']):
        """Plot solution.

        Args:
            info (:obj:`list`)
        """
        return

    def get_names(self, attr):
        """Get attribute names."""
        return getattr(self, attr).get_names()

    def copy(self):
        obj = type(self).__new__(self.__class__)
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                obj.__dict__[k] = [e.copy() if 'copy' in dir(e) else e
                                   for e in v]
            elif isinstance(v, dict):
                obj.__dict__[k] = {i: e.copy() if 'copy' in dir(e) else e
                                   for i, e in v.items()}
            elif 'copy' in dir(v):
                obj.__dict__[k] = v.copy()
            else:
                obj.__dict__[k] = v
        return obj

    def __repr__(self):
        header_start = "+-"
        header_space = "-"
        header_end = "+\n"
        l_start = "| "
        l_res = "Num. Resources: {}".format(self.resources.size())
        l_grps = "Num. Groups: {}".format(self.groups.size())
        l_pers = "Num. Periods: {}".format(self.wildfire.size())
        l_space = " "
        l_end = "|\n"
        footer_start = "+-"
        footer_space = "-"
        footer_end = "+"

        len_res = len(l_res)
        len_grps = len(l_grps)
        len_pers = len(l_pers)

        max_len = max([len_res, len_grps, len_pers]) + 4

        return header_start + header_space*max_len + header_end + \
               l_start + l_res + l_space*(max_len - len_res) + l_end + \
               l_start + l_grps + l_space*(max_len - len_grps) + l_end + \
               l_start + l_pers + l_space*(max_len - len_pers) + l_end + \
               footer_start + footer_space*max_len + footer_end
# --------------------------------------------------------------------------- #
