"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import pandas as pd

# Package modules
from firedecomp.benders import utils


# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Subproblem ------------------------------------------------------------------
class Subproblem(object):
    """Todo: Considered only selected resources."""
    def __init__(self, problem_data, min_res_penalty=1000000, relaxed=False,
                 slack=False):
        if problem_data.period_unit is False:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data
        self.min_res_penalty = min_res_penalty
        self.relaxed = relaxed
        self.slack = slack
        self.data = Expando()
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self.dual = None

        self.__build_model__()

    def __build_model__(self):
        self.model = gurobipy.Model("Subproblem")
        self.__build_data__()
        self.__build_variables__()
        self.__build_objective__()
        self.__build_constraints__()
        self.model.update()

    def __build_data__(self):
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

        self.data.Mp = self.min_res_penalty
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

        self.__update_data__()

    def __update_data__(self):
        problem_data = self.problem_data
        self.data.s_fix = {
            k: 1 if v is True else 0 for k, v in
            problem_data.resources_wildfire.get_info("start").items()}
        self.data.mu_prime_fix = {
            g: max(
                0,
                min([self.data.nMin[g, t] for t in self.data.T]) -
                sum([self.data.s_fix[i, t]
                     for i in self.data.Ig[g] for t in self.data.T]))
            for g in self.data.G}

        self.__update_work__()

    def __update_work__(self):
        """Establish work, travel and rest periods."""
        info = utils.get_start_info(self.data, self.data.s_fix)
        self.data.work = info['work']
        self.data.rest = info['rest']
        self.data.travel = info['travel']

    def __build_variables__(self):
        """Build variables."""
        data = self.data

        if self.slack is True:
            self.variables.slack = self.model.addVar(
                vtype=gurobipy.GRB.CONTINUOUS, lb=0, name="slack")

        # Wildfire
        # --------
        self.variables.mu = self.model.addVars(
            data.G, data.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0,
            name="missing_resources")

        # Other variables
        # ---------------
        self.__update_variables__()

    def __update_variables__(self):
        data = self.data

        if self.relaxed is True:
            vtype = gurobipy.GRB.CONTINUOUS
        else:
            vtype = gurobipy.GRB.BINARY

        lb = 0
        ub = 1

        s_fix = data.s_fix

        self.variables.s = self.model.addVars(
            data.I, data.T, vtype=vtype, lb=s_fix, ub=s_fix, name="start")
        self.variables.z = {
            i: sum([self.variables.s[i, t] for t in data.T]) for i in data.I}

        start = {i: min([t for t in data.T if s_fix[i, t] == 1]+[data.max_t])
                 for i in data.I}

        tr_ub = {
            (i, t): 0
            if (t < start[i]) or (self.data.rest[i, t] == 1) else 1
            for i in data.I for t in data.T}
        self.variables.tr = self.model.addVars(
            data.I, data.T, vtype=vtype, lb=lb, ub=tr_ub, name="travel")
#
        # Resources
        # ---------
        e_ub = {(i, t): sum(s_fix[i, t1] for t1 in data.T)
                for i in data.I for t in data.T}
        self.variables.e = self.model.addVars(
            data.I, data.T, vtype=vtype, lb=lb, ub=e_ub, name="end")

        # Auxiliar variables
        self.variables.u = {
            (i, t):
                self.variables.s.sum(i, data.T_int.get_names(p_max=t))
                - self.variables.e.sum(i, data.T_int.get_names(p_max=t - 1))
            for i in data.I for t in data.T}

        self.variables.w = {
            (i, t):
                data.work[i, t]*(
                        self.variables.u[i, t] - self.variables.tr[i, t])
            for i in data.I for t in data.T}

        # Wildfire
        self.variables.y = self.model.addVars(
            data.T + [data.min_t - 1], vtype=vtype, lb=lb, ub=ub,
            name="contention")
        self.variables.y[data.min_t - 1].lb = 1

    def __update_model__(self):
        """Update model. The best way found is building the model again."""
        self.__build_model__()

    def __build_objective__(self):
        """Build objective."""
        m = self.model
        data = self.data

        u = self.variables.u
        y = self.variables.y
        mu = self.variables.mu
        s = self.data.s_fix
        mu_prime = self.data.mu_prime_fix

        start_coef = {
            (i, t):
                max(data.P.values()) *
                (1 + sum([data.PR[i, t1] * data.start_work[i, t1]
                          for t1 in data.T_int.get_names(p_min=t)])/data.C[i])
            for i in data.I for t in data.T}

        self.variables.master_help_objective = \
            - sum([(data.max_t - t)*(start_coef[i, t])*s[i, t]
                   for i in data.I for t in data.T]) + \
            sum([data.Mp * mu_prime[g] for g in data.G])

        self.variables.subproblem_objective = \
            sum([data.C[i] * u[i, t] for i in data.I for t in data.T]) + \
            sum([data.NVC[t] * y[t - 1] for t in data.T]) + \
            sum([data.Mp * mu[g, t] for g in data.G for t in data.T]) + \
            0.001 * y[data.max_t]

        # Objective
        # =========
        if self.slack is False:
            m.setObjective(
                self.variables.subproblem_objective -
                self.variables.master_help_objective,
                gurobipy.GRB.MINIMIZE)
        else:
            m.setObjective(self.variables.slack, gurobipy.GRB.MINIMIZE)

    def __build_wildfire_containment_1__(self):
        data = self.data
        y = self.variables.y
        w = self.variables.w

        expr_lhs = \
            sum([data.PER[t] * y[t - 1] for t in data.T])
        expr_rhs = \
            sum([data.PR[i, t] * w[i, t] for i in data.I for t in data.T])

        if self.slack:
            expr_rhs += self.variables.slack

        self.constraints.wildfire_containment_1 = self.model.addConstr(
            expr_lhs <= expr_rhs, name='wildfire_containment_1')

    def __build_wildfire_containment_2__(self):
        data = self.data

        y = self.variables.y
        w = self.variables.w
        PER = data.PER
        PR = data.PR

        expr_lhs = {
            t: sum([PER[t1] for t1 in data.T_int.get_names(p_max=t)])*y[t - 1] -
            sum([PR[i, t1]*w[i, t1]
                 for i in data.I for t1 in data.T_int.get_names(p_max=t)])
            for t in data.T}

        expr_rhs = {t: data.M * y[t] for t in data.T}

        # if self.slack:
        #    expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.wildfire_containment_2 = self.model.addConstrs(
            (expr_lhs[t] <= expr_rhs[t] for t in data.T),
            name='wildfire_containment_2')

    def __build_end_activity__(self):
        """Travel time after end working.

        Changed respect to the original paper."""
        data = self.data

        tr = self.variables.tr
        e = self.variables.e

        expr_lhs = {(i, t): data.TRP[i]*e[i, t] for i in data.I for t in data.T}

        expr_rhs = {
            (i, t): sum([tr[i, t1]
                         for t1 in data.T_int.get_names(
                            p_min=t - data.TRP[i] + 1,  p_max=t)
                         ])
            for i in data.I for t in data.T}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.end_activity = self.model.addConstrs(
            (expr_lhs[i, t] <= expr_rhs[i, t] for i in data.I for t in data.T),
            name='end_activity')

    def __build_max_usage_periods__(self):
        data = self.data

        u = self.variables.u

        expr_lhs = {i: sum([u[i, t] for t in data.T]) for i in data.I}
        expr_rhs = {i: data.UP[i] - data.CUP[i] for i in data.I}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.max_usage_periods = self.model.addConstrs(
            (expr_lhs[i] <= expr_rhs[i]
             for i in data.I),
            name='max_usage_periods')

    def __build_non_negligence_1__(self):
        data = self.data

        y = self.variables.y
        mu = self.variables.mu
        w = self.variables.w

        expr_lhs = {
            (g, t): data.nMin[g, t] * y[t - 1] - mu[g, t]
            for g in data.G for t in data.T}
        expr_rhs = {
            (g, t): sum([w[i, t] for i in data.Ig[g]])
            for g in data.G for t in data.T}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.non_negligence_1 = self.model.addConstrs(
            (expr_lhs[g, t] <= expr_rhs[g, t]
             for g in data.G for t in data.T),
            name='non_negligence_1')

    def __build_non_negligence_2__(self):
        data = self.data

        y = self.variables.y
        w = self.variables.w

        expr_lhs = {
            (g, t): sum([w[i, t] for i in data.Ig[g]])
            for g in data.G for t in data.T}
        expr_rhs = {
            (g, t): data.nMax[g, t] * y[t - 1]
            for g in data.G for t in data.T}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.non_negligence_2 = self.model.addConstrs(
            (expr_lhs[g, t] <= expr_rhs[g, t]
             for g in data.G for t in data.T),
            name='non_negligence_2')

    def __build_logical_1__(self):
        data = self.data

        s = self.variables.s
        e = self.variables.e

        expr_lhs = {i: sum([t*s[i, t] for t in data.T]) for i in data.I}
        expr_rhs = {i: sum([t*e[i, t] for t in data.T]) for i in data.I}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.logical_1 = self.model.addConstrs(
            (expr_lhs[i] <= expr_rhs[i]
             for i in data.I),
            name='logical_1')

    def __build_logical_2__(self):
        """Changed."""
        data = self.data

        z = self.variables.z
        e = self.variables.e

        expr_lhs = {i: z[i] for i in data.I}
        expr_rhs = {i: sum([e[i, t] for t in data.T]) for i in data.I}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.logical_2 = self.model.addConstrs(
            (expr_lhs[i] <= expr_rhs[i]
             for i in data.I),
            name='logical_2')

    def __build_logical_3__(self):
        """Changed."""
        data = self.data

        u = self.variables.u
        tr = self.variables.tr

        expr_lhs = {
            (i, t): tr[i, t]
            for i in data.I for t in data.T}
        expr_rhs = {
            (i, t): u[i, t]
            for i in data.I for t in data.T}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.logical_3 = self.model.addConstrs(
            (expr_lhs[i, t] <= expr_rhs[i, t]
             for i in data.I for t in data.T),
            name='logical_3')

    def __build_logical_4__(self):
        data = self.data

        z = self.variables.z
        w = self.variables.w

        expr_lhs = {i: z[i] for i in data.I}

        expr_rhs = {i: sum([w[i, t] for t in data.T]) for i in data.I}

        # if self.slack:
        #     expr_rhs = {
        #         k: v + self.variables.slack for k, v in expr_rhs.items()}

        self.constraints.logical_4 = self.model.addConstrs(
            (expr_lhs[i] <= expr_rhs[i]
             for i in data.I),
            name='logical_4')

    def __build_constraints__(self):
        """Build constraints."""
        # Wildfire Containment
        # --------------------
        self.__build_wildfire_containment_1__()
        self.__build_wildfire_containment_2__()

        # End of Activity
        # ---------------
        self.__build_end_activity__()

        # Maximum Number of Usage Periods in a Day
        # ----------------------------------------
        self.__build_max_usage_periods__()

        # Non-Negligence of Fronts
        # ------------------------
        self.__build_non_negligence_1__()
        self.__build_non_negligence_2__()

        # Logical constraints
        # ------------------------
        self.__build_logical_1__()
        self.__build_logical_2__()
        self.__build_logical_3__()
        self.__build_logical_4__()

    def get_dual(self):
        """Get dual values of constraints."""
        return self.dual

    def get_constraint_matrix(self):
        """Get constraint matrix."""
        return pd.DataFrame(utils.get_matrix_coo(self.model)).fillna(0).T

    def get_rhs(self):
        return pd.Series(utils.get_rhs(self.model))

    def get_obj(self):
        return self.variables.subproblem_objective.getValue() - \
            self.variables.master_help_objective

    def solve(self, solver_options):
        """Solve mathematical model.

        Args:
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
        """
        if solver_options is None:
            solver_options = {'OutputFlag': 0, 'LogToConsole': 0}

        m = self.model

        # set gurobi options
        if isinstance(solver_options, dict):
            for k, v in solver_options.items():
                m.setParam(k, v)

        m.optimize()

        # Todo: check what status number return a solution
        if m.Status != 3:
            if self.relaxed is True:
                self.dual = pd.Series({c.ConstrName: c.pi
                                       for c in self.model.getConstrs()})

            if self.slack is False:
                u = {(i, t): self.variables.u[i, t].getValue() == 1
                     for i in self.data.I for t in self.data.T}
                travel = {
                    (i, t): max(self.variables.tr[i, t].x,
                                self.data.travel[i, t]*u[i, t]) == 1
                    for i in self.data.I for t in self.data.T}
                rest = {
                    (i, t): self.data.rest[i, t]*u[i, t] == 1
                    for i in self.data.I for t in self.data.T}
                end_rest = {
                    (i, t): True
                    if (rest[i, t] == 1) and (rest[i, t+1] == 0) else False
                    for i in self.data.I for t in self.data.T_int.get_names(
                            p_max=self.data.max_t-1)}
                end_rest.update({
                    (i, self.data.max_t): True
                    if rest[i, self.data.max_t] == 1 else False
                    for i in self.data.I})

                self.problem_data.resources_wildfire.update(
                    {(i, t): {
                        'travel': travel[i, t],
                        'rest': rest[i, t],
                        'end_rest': end_rest,
                        'use': u[i, t],
                        'work': self.variables.w[i, t].getValue() == 1
                    }
                     for i in self.data.I for t in self.data.T})

                self.problem_data.groups_wildfire.update(
                    {(g, t): {'number_resources': self.variables.mu[g, t].x}
                     for g in self.data.G for t in self.data.T})

                contained = {t: self.variables.y[t].x == 0 for t in self.data.T}
                contained_period = [t for t, v in contained.items()
                                    if v is True]

                if len(contained_period) > 0:
                    first_contained = min(contained_period) + 1
                else:
                    first_contained = self.max_t + 1

                self.problem_data.wildfire.update(
                    {t: {'contained': False if t < first_contained else True}
                     for t in self.data.T})
        else:
            # log.warning(config.gurobi.status_info[m.Status]['description'])
            pass
        return m.Status
# --------------------------------------------------------------------------- #
