"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import logging as log
import re
import numpy as np

# Package modules
from firedecomp.classes import solution
from firedecomp import config
from firedecomp.benders import utils


# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Master ----------------------------------------------------------------------
class Master(object):
    def __init__(self, problem_data, min_res_penalty=1000000):
        if problem_data.period_unit is False:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data
        self.data = Expando()
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self.solution = {}
        self.obj_val = None
        self.obj_val_zeta = None

        self.__build_data__(min_res_penalty)
        self.__build_model__()

    def __build_model__(self):
        self.model = gurobipy.Model("Master")
        self.__build_variables__()
        self.__build_objective__()
        self.__build_constraints__()
        self.model.update()

    def __build_data__(self, min_res_penalty):
        """Build problem data."""
        problem_data = self.problem_data
        # Sets
        self.data.I = problem_data.get_names("resources")
        self.data.G = problem_data.get_names("groups")
        self.data.T = problem_data.get_names("wildfire")
        self.data.Ig = {
            k: [e.name for e in v]
            for k, v in problem_data.groups.get_info('resources').items()}
        self.data.T_int = problem_data.wildfire

        # Parameters
        self.data.C = problem_data.resources.get_info("variable_cost")
        self.data.P = problem_data.resources.get_info("fix_cost")
        self.data.BPR = problem_data.resources.get_info("performance")
        self.data.A = problem_data.resources.get_info("arrival")
        self.data.CWP = problem_data.resources.get_info("work")
        self.data.CRP = problem_data.resources.get_info("rest")
        self.data.CUP = problem_data.resources.get_info("total_work")
        self.data.ITW = problem_data.resources.get_info("working_this_wildfire")
        self.data.IOW = problem_data.resources.get_info(
            "working_other_wildfire")
        self.data.TRP = problem_data.resources.get_info("time_between_rests")
        self.data.WP = problem_data.resources.get_info("max_work_time")
        self.data.RP = problem_data.resources.get_info("necessary_rest_time")
        self.data.UP = problem_data.resources.get_info("max_work_daily")
        self.data.PR = problem_data.resources_wildfire.get_info(
            "resource_performance")

        self.data.PER = problem_data.wildfire.get_info("increment_perimeter")
        self.data.NVC = problem_data.wildfire.get_info("increment_cost")

        self.data.nMin = problem_data.groups_wildfire.get_info("min_res_groups")
        self.data.nMax = problem_data.groups_wildfire.get_info("max_res_groups")

        self.data.Mp = min_res_penalty
        self.data.M = sum([v for k, v in self.data.PER.items()])
        self.data.min_t = int(min(self.data.T))
        self.data.max_t = int(max(self.data.T))

        variable_start = {
            (i, t): 1 if self.data.min_t == t else 0
            for i in self.data.I for t in self.data.T}
        info = utils.get_start_info(self.data, variable_start)
        self.data.start_work = info['work']
        self.data.start_travel = info['travel']
        self.data.start_rest = info['rest']
        self.data.min_resources = utils.get_minimum_resources(self.data)

    def __build_variables__(self):
        """Build variables."""
        m = self.model
        data = self.data
        # =========
        # Resources
        # ---------
        s_ub = {(i, t): 1 if (data.ITW[i] is False) or (t == data.min_t) else 0
                for i in data.I for t in data.T}
        # cr definition condition
        s_ub.update({
            (i, t): 0 for i in data.I for t in data.T
            if (data.IOW[i] == 1) and
               (t > data.min_t) and
               (t < data.min_t + data.RP[i])})

        self.variables.s = m.addVars(
            data.I, data.T, vtype=gurobipy.GRB.BINARY, ub=s_ub, name="start")

        self.variables.z = {i: self.variables.s.sum(i, '*')
                            for i in data.I}

        # Wildfire
        # --------
        # These auxiliary variables are included to give subproblem better
        # solutions.
        self.variables.mu_prime = m.addVars(
            data.G, vtype=gurobipy.GRB.CONTINUOUS,
            name="missing_resources_prime")

        # Subproblem Objective
        # --------------------
        self.variables.zeta = m.addVar(lb=0, name='zeta')

    def get_S_set(self):
        shared_variables = ["start"]
        return [
            k for k, v in self.solution.items()
            if re.match("|".join(shared_variables)+"\[.*\]", k)
            if v == 1]

    def __build_objective__(self):
        """Build objective."""
        m = self.model
        data = self.data
        s = self.variables.s
        z = self.variables.z

        zeta = self.variables.zeta
        mu_prime = self.variables.mu_prime

        start_coef = {
            (i, t):
                max(data.P.values()) *
                (1 + sum([data.PR[i, t1] * data.start_work[i, t1]
                          for t1 in data.T_int.get_names(p_min=t)])/data.C[i])
            for i in data.I for t in data.T}

        self.variables.help_obj = \
            - sum([(data.max_t - t)*(start_coef[i, t])*s[i, t]
                   for i in data.I for t in data.T]) +\
            sum([data.Mp * mu_prime[g] for g in data.G])

        self.variables.master_obj = sum([data.P[i] * z[i] for i in data.I])

        self.variables.sub_obj = zeta

        self.objective = m.setObjective(
            self.variables.master_obj +
            self.variables.help_obj +
            self.variables.sub_obj,
            gurobipy.GRB.MINIMIZE)

    def __build_constraints__(self):
        """Build variables."""
        m = self.model
        data = self.data

        # Variables
        s = self.variables.s
        z = self.variables.z
        mu_prime = self.variables.mu_prime

        # Benders constraints
        self.constraints.opt = {}
        self.constraints.opt_int = {}
        self.constraints.content_feas_int = {}
        self.constraints.feas_int = {}

        # Non-Negligence of Fronts
        # ------------------------
        self.constraints.negligence_1 = \
            m.addConstrs(
                (sum([z[i] for i in data.Ig[g]]) >=
                 min([data.nMin[g, t] for t in data.T]) - mu_prime[g]
                 for g in data.G),
                name='non-negligence_1')

        # Logical constraints
        # ------------------------
        self.constraints.logical_2 = \
            m.addConstrs(
                (z[i] <= 1
                 for i in data.I),
                name='logical_2')

        self.constraints.logical_5 = \
            m.addConstr(sum([z[i] for i in data.I]) >= data.min_resources,
                        name='logical_5')

    def add_opt_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.opt)
        self.constraints.opt[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items())
            + self.variables.zeta >= rhs,
            name='opt[{}]'.format(cut_num)
        )
        m.update()

    def add_opt_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.opt_int)
        self.constraints.opt_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items())
            + self.variables.zeta >= rhs,
            name='opt_int[{}]'.format(cut_num)
        )
        m.update()

    def add_feas_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.feas_int)
        self.constraints.feas_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items()) >= rhs,
            name='feas_int[{}]'.format(cut_num)
        )
        m.update()

    def add_contention_feas_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.content_feas_int)
        self.constraints.content_feas_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items()) >= rhs,
            name='content_feas_int[{}]'.format(cut_num)
        )
        m.update()

    def get_obj(self, zeta=True):
        """Get objective function value."""
        if zeta:
            return self.obj_val_zeta
        else:
            return self.obj_val

    def solve(self, solver_options):
        """Solve mathematical model.

        Args:
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
        """
        if solver_options is None:
            solver_options = {'OutputFlag': 0, 'LogToConsole': 0}

        m = self.model
        variables = self.variables

        # set gurobi options
        if isinstance(solver_options, dict):
            for k, v in solver_options.items():
                m.setParam(k, v)

        m.optimize()

        # Todo: check what status number return a solution
        if m.Status != 3:

            self.obj_val = self.variables.master_obj.getValue() + \
                self.variables.help_obj.getValue()
            self.obj_val_zeta = self.variables.master_obj.getValue() + \
                self.variables.help_obj.getValue() + \
                self.variables.sub_obj.x
            self.solution = {v.VarName: v.x for v in self.model.getVars()}

            # Load variables values
            self.problem_data.resources.update(
                {i: {'select': variables.z[i].getValue() == 1}
                 for i in self.data.I})

            self.problem_data.resources_wildfire.update(
                {(i, t): {
                    'start': variables.s[i, t].x == 1
                }
                 for i in self.data.I for t in self.data.T})
        else:
            # log.warning(config.gurobi.status_info[m.Status]['description'])
            pass
        return m.Status
# --------------------------------------------------------------------------- #
