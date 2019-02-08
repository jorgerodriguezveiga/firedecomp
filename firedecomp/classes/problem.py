"""Module to define input class."""

# Package modules
from firedecomp.original import model
from firedecomp.benders import benders
#from firedecomp.LR import LR
from firedecomp import config
from firedecomp import plot


# Problem ---------------------------------------------------------------------
class Problem(object):
    """Problem class."""

    def __init__(
            self, resources, wildfire, groups,
            groups_wildfire=None, resources_wildfire=None, period_unit=False,
            min_res_penalty=1000000):
        """Problem class.

        Args:
            resources
            wildfire
            groups_wildfire (:obj:`GroupsWildfire`): groups wildfire.
            resources_wildfire (:obj:`ResourcesWildfire`): resources wildfire.
            period_unit (:obj:`bool`): ``True`` if time is measured in periods.
                Defaults to ``False``.

        TODO: Add example.
        TODO: Add mathematical models as attributes.
        """
        self.resources = resources
        self.wildfire = wildfire
        self.groups = groups
        self.groups_wildfire = groups_wildfire
        self.resources_wildfire = resources_wildfire
        self.period_unit = period_unit
        self.min_res_penalty = min_res_penalty
        self.original_model = None
        self.benders_model = None
        self.branch_price_model = None
        self.lagrangian_model = None
        self.solve_status = None

        self.__build_data__()
        self.__build_period_data__()

    def get_resources(self):
        return self.resources

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

    def get_cost(self, resources=True, wildfire=True,
                 resources_penalty=True, per_period=False,
                 min_res_penalty=1000000):
        """Return objective function value."""
        cost = {t: 0 for t in self.wildfire.get_names()}
        if resources:
            cost = {k: c + sum([r.fix_cost*r.select for r in self.resources])
                    for k, c in cost.items()}
            cost = {k: c + sum([rw.resource.variable_cost*rw.use
                                for rw in self.resources_wildfire
                                if rw.get_index()[1] <= k])
                    for k, c in cost.items()}

        if wildfire:
            cost = {k: c + sum([p.increment_cost*(1-p.contained)
                                for p in self.wildfire
                                if p.get_index() <= k])
                    for k, c in cost.items()}

        if resources_penalty:
            cost = {k: c + sum([min_res_penalty * gp.num_left_resources
                                for gp in self.groups_wildfire
                                if gp.get_index()[1] <= k])
                    for k, c in cost.items()}

        if per_period is not True:
            cost = cost[max(self.wildfire.get_names())]
        return cost

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

    def update_period_data(self, max_period=None):
        self.__build_period_data__(max_period=max_period)

    def __build_period_data__(self, max_period=None):
        self.period_data = self.get_data(max_period=max_period)

    def get_data(self, max_period=None):
        if max_period is None:
            return self.data
        elif isinstance(max_period, int):
            return Data(self, max_period=max_period)
        else:
            raise TypeError(
                "max_period must be integer or None, not '{}'".format(
                    type(max_period)))

    def __build_data__(self):
        """Build problem data."""
        self.update_units()
        self.data = Data(self)

    def solve(self, method='original',
              solver_options=None, method_options=None,
              min_res_penalty=1000000,
              log_level=None):
        """Solve mathematical model.

        Args:
            method (:obj:`str`): method to solve the mathematical problem.
                Options: ``'original'``, ``'benders'``,  ``'LR'``.
                Defaults to ``'original'``.
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
            method_options (:obj:`dict`): benders method options. Default
                ``None``. If None:
                ``{'max_iters': 100, 'min_res_penalty':1000000, 'gap':0.01}``.
            min_res_penalty (:obj:`float`): positive value that penalize the
                breach of the minimum number of resources in each period.
                Defaults to ``1000000``.
            log_level (:obj:`str`): logging level. Defaults to ``None``.
        """
        self.update_units()
        if method == 'original':
            if log_level is None:
                log_level = 'WARNING'
            self.original_model = model.InputModel(
                self, min_res_penalty=min_res_penalty)
            solution = self.original_model.solve(solver_options=solver_options)
            self.solve_status = solution.model.Status
            return solution
        elif method == 'benders':
            if log_level is None:
                log_level = 'benders'

            default_benders_options = {
                'max_iters': 100,
                'mip_gap_obj': 0.01,
                'compute_feas_cuts': False,
                'solver_options_master': None,
                'solver_options_subproblem': None,
            }
            if isinstance(method_options, dict):
                default_benders_options.update(method_options)
            benders_problem = benders.Benders(
                self, **default_benders_options, log_level=log_level)
            self.solve_status = benders_problem.solve()
            return benders_problem
        elif method == 'LR':
            if log_level is None:
                log_level = 'LR'

            default_LR_options = {
                'min_res_penalty': 1000000,
                'gap': 0.01,
                'max_iters': 100,
                'max_time': 60,
                'solver_options': None,
            }
            if isinstance(method_options, dict):
                default_LR_options.update(method_options)
            LR_problem = LR.LagrangianRelaxation(
                self, **default_LR_options, log_level=log_level)
            self.solve_status = LR_problem.solve()
            return LR_problem
        else:
            raise ValueError(
                "Incorrect method '{}'. Options allowed: {}".format(
                    method, ["original"]
                ))

    def plot(self, info='scheduling'):
        """Plot solution.

        Args:
            info (:obj:`str`): What you want to plot. Options allowed:
                ``'contention'`` or ``'scheduling'``.
        """
        if info == 'contention':
            plot.solution.plot_contention(self)
        elif info == 'scheduling':
            plot.solution.plot_scheduling(self)

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


# Data ------------------------------------------------------------------------
class Data(object):
    """Data class."""

    def __init__(self, problem, max_period=float('inf')):
        # Sets
        self.I = problem.get_names("resources")
        self.G = problem.get_names("groups")
        self.T = [t for t in problem.get_names("wildfire") if t <= max_period]
        self.Ig = {
            k: [e.name for e in v]
            for k, v in problem.groups.get_info('resources').items()}

        # Parameters
        self.C = problem.resources.get_info("variable_cost")
        self.P = problem.resources.get_info("fix_cost")
        self.BPR = problem.resources.get_info("performance")
        self.A = problem.resources.get_info("arrival")
        self.CWP = problem.resources.get_info("work")
        self.CRP = problem.resources.get_info("rest")
        self.CUP = problem.resources.get_info("total_work")
        self.ITW = problem.resources.get_info("working_this_wildfire")
        self.IOW = problem.resources.get_info("working_other_wildfire")
        self.TRP = problem.resources.get_info("time_between_rests")
        self.WP = problem.resources.get_info("max_work_time")
        self.RP = problem.resources.get_info("necessary_rest_time")
        self.UP = problem.resources.get_info("max_work_daily")
        self.PR = {
            k: v for k, v in problem.resources_wildfire.get_info(
            "resource_performance").items()
            if k[1] <= max_period
        }

        self.PER = {
            t: v
            for t, v in problem.wildfire.get_info("increment_perimeter").items()
            if t <= max_period
        }
        self.NVC = {
            t: v for t, v in problem.wildfire.get_info("increment_cost").items()
            if t <= max_period
        }

        self.nMin = {
            k: v
            for k, v in problem.groups_wildfire.get_info(
            "min_res_groups").items()
            if k[1] <= max_period
        }
        self.nMax = {
            k: v
            for k, v in problem.groups_wildfire.get_info(
            "max_res_groups").items()
            if k[1] <= max_period
        }

        self.Mp = problem.min_res_penalty
        self.M = sum([v for k, v in self.PER.items()])
        self.min_t = int(min(self.T))
        self.max_t = int(max(self.T))

    def T_int(self, p_min=None, p_max=None):
        if p_min is None:
            p_min = self.min_t
        else:
            p_min = max(p_min, self.min_t)

        if p_max is None:
            p_max = self.max_t
        else:
            p_max = min(p_max, self.max_t)
        return [t for t in range(int(p_min), int(p_max+1))]
# --------------------------------------------------------------------------- #
