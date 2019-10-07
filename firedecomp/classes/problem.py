"""Module to define input class."""

# Python package
import time

# Package modules
from firedecomp.original import model as _model
from firedecomp.fix_work import fix_work
from firedecomp.original import model as _model
from firedecomp.benders import benders
from firedecomp.LR import LR
from firedecomp.AL import AL
from firedecomp import config
from firedecomp import plot
import firedecomp.branchprice.model_original as scip_model
import firedecomp.branchprice.benders_scip as scip
from firedecomp.config import scip as scip_status

import time

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
        self.original_scip_model = None
        self.fix_work_model = None
        self.benders_scip_model = None
        self.gcg_scip_model = None
        self.LR_model = None
        self.AL_model = None
        self.solve_status = None
        self.mipgap = None
        self.constrvio = None
        self.time = None
        self.solve_time = None

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

    def check_feasibility(self):
        """Check problem feasibility.

        Todo: Check problem feasibility.
        """
        pass

    def get_slack_wildfire_containment_1(self):
        """Get slack of the wildfire containment 1 constraints.
        """
        self.update_units()
        return self.resources_wildfire.get_performance() - \
            self.wildfire.get_contention_perimeter()

    def get_slack_wildfire_containment_2(self):
        """Get slack of the wildfire containment 2 constraints.
        """
        period_performance = {
            p.get_index():
                p.__resource_period__.get_performance() -
                p.get_increment_perimeter()
            for p in self.wildfire}
        cum = 0
        for k, v in period_performance.items():
            cum += v
            period_performance[k] = cum
        return period_performance

    def get_num_selected_resources(self) -> int:
        """Get the number of selected resources."""
        return sum([r.select for r in self.resources])

    def get_contention_period(self) -> int:
        """Get the contention period."""
        return sum([p.contained is False for p in self.wildfire])

    def get_cost(self, resources=True, wildfire=True,
                 resources_penalty=True, per_period=False):
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
            cost = {k: c + sum([self.min_res_penalty * gp.num_left_resources
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

    def get_variables_solution(self)->dict:
        """Get variable solution as a dictionary."""
        solution_dict = {
            's': {k: float(v) for k, v in self.resources_wildfire.get_info("start").items()},
            'tr': {k: float(v) for k, v in self.resources_wildfire.get_info("travel").items()},
            'r': {k: float(v) for k, v in self.resources_wildfire.get_info("rest").items()},
            'e': {k: float(v) for k, v in self.resources_wildfire.get_info("end").items()},
            'er': {k: float(v) for k, v in self.resources_wildfire.get_info("end_rest").items()},
            'mu': {k: float(v) for k, v in self.groups_wildfire.get_info("num_left_resources").items()},
            'y': {k: float(v) for k, v in self.wildfire.get_info("contained").items()}
        }
        return solution_dict

    def solve(self, method='original',
              original_options=None, original_scip_options=None,
              fix_work_options=None, benders_scip_options=None,
              gcg_scip_options=None, LR_options=None, AL_options=None,
              min_res_penalty=1000000,
              log_level=None):
        """Solve mathematical model.

        Args:
            method (:obj:`str`): method to solve the mathematical problem.
                Options: ``'original'``, ``'fix_work'``. Defaults to
                ``'original'``.
            original_options (:obj:`dict`): gurobi options. Default ``None``.
                If ``None`` defaults options.
            original_scip_options(:obj:`dict`): scip options. Defaults to
                ``None``. If ``None`` defaults options.
            fix_work_options (:obj:`dict`): fix work method options. Default
                ``None``. If ``None`` default options.
            benders_scip_options(:obj:`dict`): benders_scip options. Defaults to
                ``None``. If ``None`` defaults options.
            gcg_scip_options(:obj:`dict`): gcg_scip options. Defaults to
                ``None``. If ``None`` defaults options.
            min_res_penalty (:obj:`float`): positive value that penalize the
                breach of the minimum number of resources in each period.
                Defaults to ``1000000``.
            log_level (:obj:`str`): logging level. Defaults to ``None``.
        """
        self.update_units()
        self.time = None
        self.solve_time = None
        start_time = time.time()
        if method == 'original':
            valid_constraints = []
            solver_options = {}
            if isinstance(original_options, dict):
                if 'valid_constraints' in original_options:
                    valid_constraints = original_options['valid_constraints']

                if 'solver_options' in original_options:
                    solver_options = original_options['solver_options']

            self.original_model = _model.InputModel(
                self, min_res_penalty=min_res_penalty,
                valid_constraints=valid_constraints)
            solution = self.original_model.solve(
                solver_options=solver_options)
            self.solve_status = solution.model.Status
            if solution.model.SolCount >= 1:
                self.constrvio = solution.model.constrvio
                self.mipgap = solution.model.mipgap
            else:
                self.constrvio = None
                self.mipgap = None
            self.solve_time = solution.model.Runtime
        elif method == 'original_scip':
            if original_scip_options is None:
                original_scip_options = {}

            self.original_scip_model, self.solve_status = scip.solve_original(
                self, solver_options=original_scip_options)

            # Get information
            model = self.original_scip_model.model
            self.solve_status = scip_status.status[model.getStatus()]
            try:
                self.constrvio = 0
                for c in model.getConss():
                    try:
                        viol = max(0, - model.getSlack(c))
                    except Exception:
                        viol = 0
                    if viol > self.constrvio:
                        self.constrvio = viol
                self.mipgap = model.getGap()
            except Exception:
                self.constrvio = None
                self.mipgap = None
            self.solve_time = model.getSolvingTime()

        elif method == 'fix_work':
            if log_level is None:
                log_level = 'fix_work'

            if fix_work_options is None:
                fix_work_options = {}
            self.fix_work_model = fix_work.FixWorkAlgorithm(
                self, **fix_work_options, log_level=log_level)
            self.solve_status = self.fix_work_model.solve()
            if self.fix_work_model.master.model.SolCount >= 1:
                self.constrvio = self.fix_work_model.master.model.constrvio
                self.mipgap = self.fix_work_model.master.model.mipgap
            else:
                self.constrvio = None
                self.mipgap = None
            self.solve_time = self.fix_work_model.runtime

        elif method == 'benders_scip':
            if benders_scip_options is None:
                benders_scip_options = {}

            try:
                self.benders_scip_model, self.solve_status = scip.solve_benders(
                    self, solver_options=benders_scip_options)

                # Get information
                model = self.benders_scip_model.model
                self.solve_status = scip_status.status[model.getStatus()]
                # Compute constraint violation
                self.constrvio = 0
                for c in model.getConss():
                    try:
                        viol = max(0, - model.getSlack(c))
                    except Exception:
                        viol = 0
                    if viol > self.constrvio:
                        self.constrvio = viol
                self.mipgap = model.getGap()
                self.solve_time = model.getSolvingTime()

            except Exception as e:
                print(e)
                self.solve_status = None
                self.constrvio = None
                self.mipgap = None
                self.solve_time = None
        elif method == 'gcg_scip':
            if gcg_scip_options is None:
                gcg_scip_options = {}

            try:
                # Solving the problem with GCG via call to system
                self.solve_status, self.solve_time, self.mipgap, self.constrvio = scip.solve_GCG(
                    self, model_name='fireproblem', solver_options=gcg_scip_options)
            except Exception as e:
                print(e)
                self.solve_status = None
                self.solve_time = None
                self.mipgap = None
                self.constrvio = None

            # There is no model
            self.gcg_scip_model = None

        elif method == 'LR':
            if log_level is None:
                log_level = 'LR'

            if LR_options is None:
                LR_options = {
                    'min_res_penalty': 1000000,
                    'gap': 0.0,
                    'max_iters': 100,
                    'max_time': 60
                }

            LR_problem = LR.LagrangianRelaxation(
                self, **LR_options, log_level=log_level)
            self.solve_status = LR_problem.solve()
            self.LR_model = LR_problem

        elif method == 'AL':
            if log_level is None:
                log_level = 'AL'

            if AL_options is None:
                AL_options = {
                    'min_res_penalty': 1000000,
                    'gap': 0.0,
                    'max_iters': 100,
                    'max_time': 60
                }

            AL_problem = AL.AugmentedLagrangian(
                self, **AL_options, log_level=log_level)
            self.solve_status = AL_problem.solve()
            self.AL_model = AL_problem

        else:
            raise ValueError(
                "Incorrect method '{}'. Options allowed: {}".format(
                    method,
                    ["original", "original_scip", "fix_work", "benders_scip",
                     "gcg_scip"]
                ))
        self.time = time.time() - start_time

    def get_solution_info(self):
        """Get solution information."""
        try:
            res_cost = self.get_cost(
                    resources=True, wildfire=False, resources_penalty=False)
        except TypeError:
            res_cost = None

        try:
            wildfire_cost = self.get_cost(
                    resources=False, wildfire=True, resources_penalty=False)
        except TypeError:
            wildfire_cost = None

        try:
            res_penalty = self.get_cost(
                    resources=False, wildfire=False, resources_penalty=True)
        except TypeError:
            res_penalty = None

        if ((res_cost is not None) and (wildfire_cost is not None) and (res_penalty is not None)):
            objfun = res_cost + wildfire_cost + res_penalty
        else:
            objfun = None

        try:
            selected_resources = self.get_num_selected_resources()
        except TypeError:
            selected_resources = None

        if selected_resources:
            contention_period = self.get_contention_period()
        else:
            contention_period = None

        if objfun is not None and self.mipgap is not None:
            mipgapabs = self.mipgap*objfun
        else:
            mipgapabs = None

        return {
            'obj_fun': objfun,
            'res_cost': res_cost,
            'wildfire_cost': wildfire_cost,
            'resources_penalty': res_penalty,
            'mipgap': self.mipgap,
            'mipgapabs': mipgapabs,
            'constrvio': self.constrvio,
            'status': self.solve_status,
            'solve_time': self.solve_time,
            'elapsed_time': self.time,
            'selected_resources': selected_resources,
            'contention_period': contention_period
        }

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

    def get_initial_sol(self) -> bool:
        """Update the model with an initial solution.

        Return:
            If initial solution is feasible return True otherwise False.
        """
        return fix_work.utils.get_initial_sol(self)

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

    def slack_wildfire_containment_1(self):
        return sum([self.data.PR[i,t] * self.resources_wildfire[i, t].work
                    for i in self.data.I for t in self.data.T]) - \
            sum([self.data.PER[t] * (not self.wildfire[t].contained)
                 for t in self.data.T])

    def slack_wildfire_containment_2(self):
        data = self.data
        return {
            t:
                data.M * (not self.wildfire[t+1].contained) -
                sum([data.PER[t1]
                     for t1 in data.T_int(p_max=t)]) *
                (not self.wildfire[t].contained) +
                sum([data.PR[i, t1] * self.resources_wildfire[i, t1].work
                     for i in data.I for t1 in data.T_int(p_max=t)])
            if t < max(data.T) else
                sum([data.PER[t1]
                     for t1 in data.T_int(p_max=t)]) *
                (not self.wildfire[t].contained) +
                sum([data.PR[i, t1] * self.resources_wildfire[i, t1].work
                     for i in data.I for t1 in data.T_int(p_max=t)])
            for t in data.T
        }

    def slack_start_activity_1(self):
        data = self.data
        return {(i, t):
                    sum([self.resources_wildfire[i, t1].travel
                         for t1 in data.T_int(p_max=t)]) -
                    self.data.A[i] * self.resources_wildfire[i, t].work
         for i in data.I for t in data.T}

    def slack_start_activity_2(self):
        data = self.data
        return {
            i:
                data.max_t*self.resources[i].select - self.resources_wildfire[i, data.min_t].start -
                sum([(data.max_t + 1)*self.resources_wildfire[i, t].start
                    for t in data.T_int(p_min=data.min_t+1)])
            for i in data.I if data.ITW[i] == True
        }

    def slack_start_activity_3(self):
        data = self.data
        return {
            i:
                self.resources[i].select -
                sum([self.resources_wildfire[i, t].start for t in data.T])
            for i in data.I if data.ITW[i] == False
        }

    def slack_end_activity(self):
        data = self.data
        return {
            (i, t):
                sum([self.resources_wildfire[i, t1].travel
                     for t1 in data.T_int(p_min=t-data.TRP[i]+1, p_max=t)]) -
                data.TRP[i]*self.resources_wildfire[i, t].end
            for i in data.I for t in data.T
        }

    def get_cr(self):
        data = self.data
        cr = {(i, t):
            sum([
                (t + 1 - t1) * self.resources_wildfire[i, t1].start
                - (t - t1) * self.resources_wildfire[i, t1].end
                - self.resources_wildfire[i, t1].rest
                - data.WP[i] * self.resources_wildfire[i, t1].end_rest
                for t1 in data.T_int(p_max=t)])
            for i in data.I for t in data.T
            if not data.ITW[i] and not data.IOW[i]
        }
        cr.update({
            (i, t):
                (t+data.CWP[i]-data.CRP[i]) *
                self.resources_wildfire[i, data.min_t].start
                + sum([
                    (t + 1 - t1 + data.WP[i]) *
                    self.resources_wildfire[i, t1].start
                    for t1 in data.T_int(p_min=data.min_t + 1, p_max=t)])
                - sum([
                    (t - t1) * self.resources_wildfire[i, t1].end
                    + self.resources_wildfire[i, t1].rest
                    + data.WP[i] * self.resources_wildfire[i, t1].end_rest
                    for t1 in data.T_int(p_max=t)])
            for i in data.I for t in data.T
            if data.ITW[i] or data.IOW[i]})
        return cr

    def slack_break_1_lb(self):
        data = self.data
        cr = self.get_cr()
        return {(i, t): cr[i, t] for i in data.I for t in data.T}

    def slack_break_1_ub(self):
        data = self.data
        cr = self.get_cr()
        return {(i, t): data.WP[i] - cr[i, t]
                for i in data.I for t in data.T}

    def slack_break_2(self):
        data = self.data
        return {
            (i, t):
                sum([self.resources_wildfire[i, t1].end_rest
                     for t1 in data.T_int(p_min=t, p_max=t+data.RP[i]-1)]) -
                self.resources_wildfire[i, t].rest
            for i in data.I for t in data.T
        }

    def slack_break_3(self):
        data = self.data
        res = self.resources_wildfire
        return {
            (i, t):
                sum([res[i, t1].rest for t1 in data.T_int(
                    p_min=t - data.RP[i] + 1, p_max=t)]) -
                data.RP[i] * res[i, t].end_rest
            if t >= data.min_t - 1 + data.RP[i] else
                data.CRP[i] * res[i, data.min_t].start +
                sum([res[i, t1].rest for t1 in data.T_int(p_max=t)]) -
                data.RP[i] * res[i, t].end_rest
            for i in data.I for t in data.T
        }

    def slack_break_4(self):
        data = self.data
        res = self.resources_wildfire
        return {
            (i, t):
                sum([res[i, t1].rest + res[i, t1].travel
                     for t1 in data.T_int(p_min=t-data.TRP[i],
                                          p_max=t+data.TRP[i])]) -
                len(data.T_int(p_min=t-data.TRP[i],
                               p_max=t+data.TRP[i])) * res[i, t].rest
            for i in data.I for t in data.T
        }
        # return {
        #     (i, t):
        #         sum([1-res[i, t1].work
        #              for t1 in data.T_int(p_min=t-data.TRP[i],
        #                                   p_max=t+data.TRP[i])]) -
        #         len(data.T_int(p_min=t-data.TRP[i],
        #                        p_max=t+data.TRP[i])) * res[i, t].rest
        #     for i in data.I for t in data.T
        # }

    def slack_max_usage_periods(self):
        data = self.data
        res = self.resources_wildfire
        return {
            i: data.UP[i] - data.CUP[i] - sum([res[i, t].use for t in data.T])
            for i in data.I
        }

    def slack_non_negligence_1(self):
        data = self.data
        res = self.resources_wildfire
        return {
            (g, t):
                sum([res[i, t].work for i in data.Ig[g]]) -
                data.nMin[g, t] * (not self.wildfire[t].contained) +
                self.groups_wildfire[g, t].num_left_resources
            for g in data.G for t in data.T
        }

    def slack_non_negligence_2(self):
        data = self.data
        res = self.resources_wildfire
        return {
            (g, t):
                data.nMax[g, t] * (not self.wildfire[t].contained) -
                sum([res[i, t].work for i in data.Ig[g]])
            for g in data.G for t in data.T
        }

    def slack_logical_1(self):
        data = self.data
        res = self.resources_wildfire
        return {
            i: sum([t * res[i, t].end for t in data.T]) - sum(
                [t * res[i, t].start for t in data.T])
            for i in data.I
        }

    def slack_logical_2(self):
        data = self.data
        res = self.resources_wildfire
        return {
            i: 1 - sum([res[i, t].end for t in data.T])
            for i in data.I
        }

    def slack_logical_3(self):
        data = self.data
        res = self.resources_wildfire
        return {
            (i, t): res[i, t].use - res[i, t].rest - res[i, t].travel
            for i in data.I for t in data.T
        }

    def slack_logical_4(self):
        data = self.data
        res = self.resources_wildfire
        return {
            i: sum([res[i, t].work for t in data.T]) - self.resources[i].select
            for i in data.I
        }

    def get_infeas_constraints(self, zero=1.0e-12):
        infeas = {}

        infeas['wildfire_containment_1'] = {'': self.slack_wildfire_containment_1()}
        infeas['wildfire_containment_2'] = self.slack_wildfire_containment_2()
        infeas['start_activity_1'] = self.slack_start_activity_1()
        infeas['start_activity_2'] = self.slack_start_activity_3()
        infeas['end_activity'] = self.slack_end_activity()
        infeas['break_1_lb'] = self.slack_break_1_lb()
        infeas['break_1_ub'] = self.slack_break_1_ub()
        infeas['break_2'] = self.slack_break_2()
        infeas['break_3'] = self.slack_break_3()
        infeas['break_4'] = self.slack_break_4()
        infeas['max_usage_periods'] = self.slack_max_usage_periods()
        infeas['non_negligence_1'] = self.slack_non_negligence_1()
        infeas['non_negligence_2'] = self.slack_non_negligence_2()
        infeas['logical_1'] = self.slack_logical_1()
        infeas['logical_2'] = self.slack_logical_2()
        infeas['logical_3'] = self.slack_logical_3()
        infeas['logical_4'] = self.slack_logical_4()
        infeas = {
            k: {k1: v1 for k1, v1 in v.items() if v1 < -zero}
            for k, v in infeas.items()
        }
        return {k: v for k, v in infeas.items() if len(v) > 0}
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
