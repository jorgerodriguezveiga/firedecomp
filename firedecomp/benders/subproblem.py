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
        if problem_data.period_unit is not True:
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

        self.data = self.problem_data.data
        self.__build_model__()

    def __build_model__(self):
        self.model = gurobipy.Model("Subproblem")
        self.__update_data__()
        self.__build_variables__()
        self.__build_objective__()
        self.__build_constraints__()
        self.model.update()

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
        self.variables.z = {i: sum([s_fix[i, t] for t in data.T])
                            for i in data.I}

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
        z = self.variables.z

        self.variables.fix_cost_resources = sum(
            [data.P[i] * z[i] for i in data.I])
        self.variables.variable_cost_resources = sum(
            [data.C[i] * u[i, t] for i in data.I for t in data.T])
        self.variables.wildfire_cost = sum(
            [data.NVC[t] * y[t - 1] for t in data.T])
        self.variables.law_cost = sum(
            [data.Mp * mu[g, t] for g in data.G for t in data.T])

        self.variables.subproblem_objective = \
            self.variables.fix_cost_resources + \
            self.variables.variable_cost_resources + \
            self.variables.wildfire_cost + \
            self.variables.law_cost + \
            0.001 * y[data.max_t]

        # Objective
        # =========
        if self.slack is not True:
            m.setObjective(
                self.variables.subproblem_objective,
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

    def get_obj(self, resources_fix=True, resources_variable=True,
                wildfire=True, law=True):
        obj_val = 0

        if resources_fix:
            obj_val += self.variables.fix_cost_resources

        if resources_variable:
            obj_val += self.variables.variable_cost_resources.getValue()

        if wildfire:
            obj_val += self.variables.wildfire_cost.getValue()

        if law:
            obj_val += self.variables.law_cost.getValue()

        return obj_val

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

            if self.slack is not True:
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
                    {(g, t): {'num_left_resources': self.variables.mu[g, t].x}
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
