"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import logging as log

# Package modules
from firedecomp.classes import solution
from firedecomp import config


# InputModel ------------------------------------------------------------------
class InputModel(object):
    def __init__(self, problem_data, relaxed=False, min_res_penalty=1000000,
                 valid_constraints=None):
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data
        self.relaxed = relaxed

        # Sets
        self.I = problem_data.get_names("resources")
        self.G = problem_data.get_names("groups")
        self.T = problem_data.get_names("wildfire")
        self.Ig = {
            k: [e.name for e in v]
            for k, v in problem_data.groups.get_info('resources').items()}
        self.T_int = problem_data.wildfire

        # Parameters
        self.C = problem_data.resources.get_info("variable_cost")
        self.P = problem_data.resources.get_info("fix_cost")
        self.BPR = problem_data.resources.get_info("performance")
        self.A = problem_data.resources.get_info("arrival")
        self.CWP = problem_data.resources.get_info("work")
        self.CRP = problem_data.resources.get_info("rest")
        self.CUP = problem_data.resources.get_info("total_work")
        self.ITW = problem_data.resources.get_info("working_this_wildfire")
        self.IOW = problem_data.resources.get_info("working_other_wildfire")
        self.TRP = problem_data.resources.get_info("time_between_rests")
        self.WP = problem_data.resources.get_info("max_work_time")
        self.RP = problem_data.resources.get_info("necessary_rest_time")
        self.UP = problem_data.resources.get_info("max_work_daily")
        self.PR = problem_data.resources_wildfire.get_info(
            "resource_performance")

        self.PER = problem_data.wildfire.get_info("increment_perimeter")
        self.NVC = problem_data.wildfire.get_info("increment_cost")

        self.nMin = problem_data.groups_wildfire.get_info("min_res_groups")
        self.nMax = problem_data.groups_wildfire.get_info("max_res_groups")

        self.Mp = min_res_penalty
        self.M = sum([v for k, v in self.PER.items()])
        self.min_t = int(min(self.T))
        self.max_t = int(max(self.T))
        self.valid_constraints = valid_constraints
        self.sizey = self.T + [self.min_t-1]
        self.model = self.__get_model__()

    def set_model(self, model):
        self.model = solution.Solution(model, dict(s=self.s, tr=self.tr, r=self.r, er=self.er, e=self.e, u=self.u, w=self.w, z=self.z, cr=self.cr, y=self.y, mu=self.mu))


    def __get_model__(self):
        return self.model(self, relaxed=self.relaxed,
                     valid_constraints=self.valid_constraints)

    def solve(self, solver_options=None):
        """Solve mathematical model.

        Args:
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
        """
        if solver_options is None:
            solver_options = {'OutputFlag': 0, 'LogToConsole':0}

        m = self.model

        # set gurobi options
        if isinstance(solver_options, dict):
            for k, v in solver_options.items():
                m.model.setParam(k, v)

        m.model.optimize()

        # Check if exist a solution
        if m.model.SolCount >= 1 and m.model.Status != 3:
            if self.relaxed is not True:
                # Load variables values
                self.problem_data.resources.update_select(
                    {i: round(m.variables.z[i].getValue()) == 1
                     for i in self.I})

                self.problem_data.resources_wildfire.update(
                    {(i, t): {
                        'start': round(m.variables.s[i, t].x) == 1,
                        'travel': round(m.variables.tr[i, t].x) == 1,
                        'rest': round(m.variables.r[i, t].x) == 1,
                        'end_rest': round(m.variables.er[i, t].x) == 1,
                        'end': round(m.variables.e[i, t].x) == 1,
                        'use': round(m.variables.u[i, t].getValue()) == 1,
                        'work': round(m.variables.w[i, t].getValue()) == 1
                    }
                     for i in self.I for t in self.T})

                self.problem_data.groups_wildfire.update(
                    {(g, t): {'num_left_resources': m.variables.mu[g, t].x}
                     for g in self.G for t in self.T})

                contained = {t: round(m.variables.y[t].x) == 0 for t in self.T}
                contained_period = [t for t, v in contained.items()
                                    if v is True]

                if len(contained_period) > 0:
                    first_contained = min(contained_period) + 1
                else:
                    first_contained = self.max_t + 1

                self.problem_data.wildfire.update(
                    {t: {'contained': False if t < first_contained else True}
                     for t in self.T})
        else:
            log.warning("No solutions found.")
            log.warning(
                config.gurobi.status_info[m.model.Status]['description'])

        return m
# --------------------------------------------------------------------------- #


    # model -----------------------------------------------------------------------
    def model(self, data, relaxed=False, slack_containment=False, valid_constraints=None,
              slack_penalty=1000000000000):
        """Wildfire suppression model.

        Args:
            data (:obj:`firedecomp.model.InputModel`): problem data.
            relaxed (:obj:`bool`): if True variables will be continuous. Defaults
                to False.
            slack_containment (:obj:`bool`): if True add slack variable to wildfire
                containment constraints. Defaults to False.
            slack_penalty (:obj:`float`): slack containment penalty.
        """
        self.m = gurobipy.Model("wildfire_supression")

        if relaxed is True:
            vtype = gurobipy.GRB.CONTINUOUS
            lb = 0
            ub = 1
        else:
            vtype = gurobipy.GRB.BINARY
            lb = 0
            ub = 1

        if valid_constraints is None:
            valid_constraints = ['contention', 'work', 'max_obj']

        # Variables
        # =========
        # Resources
        # ---------
        self.s = self.m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="start")
        self.tr = self.m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="travel")
        self.r = self.m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="rest")
        self.er = self.m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="end_rest")
        self.e = self.m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="end")

        # Auxiliar variables
        self.u = {
            (i, t):
                self.s.sum(i, data.T_int.get_names(p_max=t))
                - self.e.sum(i, data.T_int.get_names(p_max=t-1))
            for i in data.I for t in data.T}
        self.w = {(i, t): self.u[i, t] - self.r[i, t] - self.tr[i, t] for i in data.I for t in data.T}
        self.z = {i: self.e.sum(i, '*') for i in data.I}

        self.cr = {(i, t):
              sum([
                  (t+1-t1)*self.s[i, t1]
                  - (t-t1)*self.e[i, t1]
                  - self.r[i, t1]
                  - data.WP[i]*self.er[i, t1]
                  for t1 in data.T_int.get_names(p_max=t)])
              for i in data.I for t in data.T
              if not data.ITW[i] and not data.IOW[i]}

        self.cr.update({
            (i, t):
                (t+data.CWP[i]-data.CRP[i]) * self.s[i, data.min_t]
                + sum([
                    (t + 1 - t1 + data.WP[i]) * self.s[i, t1]
                    for t1 in data.T_int.get_names(p_min=data.min_t + 1, p_max=t)])
                - sum([
                    (t - t1) * self.e[i, t1]
                    + self.r[i, t1]
                    + data.WP[i] * self.er[i, t1]
                    for t1 in data.T_int.get_names(p_max=t)])
            for i in data.I for t in data.T
            if data.ITW[i] or data.IOW[i]})

        # Wildfire
        # --------
        self.y = self.m.addVars(data.T + [data.min_t-1], vtype=vtype, lb=lb, ub=ub, name="contention")
        self.mu = self.m.addVars(data.G, data.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0, name="missing_resources")

        if slack_containment:
            slack_cont1 = self.m.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, name="slack_containment_1")
            slack_cont2 = self.m.addVars(data.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0, name="slack_containment_2")

        # Objective
        # =========
        self.obj_expression = \
            sum([data.C[i]*self.u[i, t] for i in data.I for t in data.T]) + \
            sum([data.P[i] * self.z[i] for i in data.I]) + \
            sum([data.NVC[t] * self.y[t-1] for t in data.T]) + \
            sum([data.Mp*self.mu[g, t] for g in data.G for t in data.T]) + \
            0.001*self.y[data.max_t]

        if slack_containment:
            self.obj_expression += \
                slack_penalty * slack_cont1 + \
                sum([slack_penalty * slack_cont2[t] for t in data.T])

        self.m.setObjective(self.obj_expression, gurobipy.GRB.MINIMIZE)

        # Constraints
        # ===========

        # Valid constraints
        # -----------------

        if 'contention' in valid_constraints:
            expr_lhs = {t: self.y[t] for t in data.T}
            expr_rhs = {t: self.y[t - 1] for t in data.T}

            self.m.addConstrs(
                (expr_lhs[t] <= expr_rhs[t] for t in data.T),
                name='valid_constraint_contention'
            )

        if 'work' in valid_constraints:
            expr_lhs = {(i, t): self.w[i, t] for i in data.I for t in data.T}
            expr_rhs = {t: self.y[t - 1] for t in data.T}

            self.m.addConstrs(
                (expr_lhs[i, t] <= expr_rhs[t] for i in data.I for t in data.T),
                name='valid_constraint_contention'
            )

        # Wildfire Containment
        # --------------------
        self.m.addConstr(self.y[data.min_t-1] == 1, name='start_no_contained')

        if slack_containment:
            wildfire_containment_1_expr = \
                sum([data.PER[t] * self.y[t - 1] for t in data.T]) <= \
                sum([data.PR[i, t] * self.w[i, t] for i in data.I for t in data.T]) + \
                slack_cont1
        else:
            wildfire_containment_1_expr = \
                sum([data.PER[t] * self.y[t - 1] for t in data.T]) <= \
                sum([data.PR[i, t] * self.w[i, t] for i in data.I for t in data.T])

        self.m.addConstr(wildfire_containment_1_expr, name='wildfire_containment_1')

        if slack_containment:
            wildfire_containment_2_expr = (
                data.M*self.y[t] + slack_cont2[t] >=
                sum([data.PER[t1]
                     for t1 in data.T_int.get_names(p_max=t)])*self.y[t-1] -
                sum([data.PR[i, t1]*self.w[i, t1]
                     for i in data.I for t1 in data.T_int.get_names(p_max=t)])
                for t in data.T)
        else:
            wildfire_containment_2_expr = (
                data.M*self.y[t] >=
                sum([data.PER[t1]
                     for t1 in data.T_int.get_names(p_max=t)])*self.y[t-1] -
                sum([data.PR[i, t1]*self.w[i, t1]
                     for i in data.I for t1 in data.T_int.get_names(p_max=t)])
                for t in data.T)

        self.m.addConstrs(wildfire_containment_2_expr, name='wildfire_containment_2')

        # Start of activity
        # -----------------
        self.m.addConstrs(
            (data.A[i]*self.w[i, t] <=
             sum([self.tr[i, t1] for t1 in data.T_int.get_names(p_max=t)])
             for i in data.I for t in data.T),
            name='start_activity_1')

        self.m.addConstrs(
            (self.s[i, data.min_t] +
             sum([(data.max_t + 1)*self.s[i, t]
                  for t in data.T_int.get_names(p_min=data.min_t+1)]) <=
             data.max_t*self.z[i]
             for i in data.I if data.ITW[i] == True),
            name='start_activity_2')

        self.m.addConstrs(
            (sum([self.s[i, t] for t in data.T]) <= self.z[i]
             for i in data.I if data.ITW[i] == False),
            name='start_activity_3')

        # End of Activity
        # ---------------
        self.m.addConstrs(
            (sum([self.tr[i, t1] for t1 in data.T_int.get_names(p_min=t-data.TRP[i]+1, p_max=t)
                  ]) >= data.TRP[i]*self.e[i, t]
             for i in data.I for t in data.T),
            name='end_activity')

        # Breaks
        # ------
        self.m.addConstrs(
            (0 <= self.cr[i, t]
             for i in data.I for t in data.T),
            name='Breaks_1_lb')

        self.m.addConstrs(
            (self.cr[i, t] <= data.WP[i]
             for i in data.I for t in data.T),
            name='Breaks_1_ub')

        self.m.addConstrs(
            (self.r[i, t] <= sum([self.er[i, t1]
                             for t1 in data.T_int.get_names(p_min=t,
                                                            p_max=t+data.RP[i]-1)])
             for i in data.I for t in data.T),
            name='Breaks_2')

        self.m.addConstrs(
            (sum([
                self.r[i, t1]
                for t1 in data.T_int.get_names(p_min=t-data.RP[i]+1, p_max=t)]) >=
             data.RP[i]*self.er[i, t]
             if t >= data.min_t - 1 + data.RP[i] else
             data.CRP[i]*self.s[i, data.min_t] +
             sum([self.r[i, t1] for t1 in data.T_int.get_names(p_max=t)]) >=
             data.RP[i]*self.er[i, t]
             for i in data.I for t in data.T),
            name='Breaks_3')

        self.m.addConstrs(
            (sum([self.r[i, t1]+self.tr[i, t1]
                  for t1 in data.T_int.get_names(p_min=t-data.TRP[i],
                                                 p_max=t+data.TRP[i])]) >=
             len(data.T_int.get_names(p_min=t-data.TRP[i],
                                      p_max=t+data.TRP[i]))*self.r[i, t]
             for i in data.I for t in data.T),
            name='Breaks_4')

        # Maximum Number of Usage Periods in a Day
        # ----------------------------------------
        self.m.addConstrs(
            (sum([self.u[i, t] for t in data.T]) <= data.UP[i] - data.CUP[i]
             for i in data.I),
            name='max_usage_periods')

        # Non-Negligence of Fronts
        # ------------------------
        self.m.addConstrs(
            (sum([self.w[i, t] for i in data.Ig[g]]) >= data.nMin[g, t]*self.y[t-1] - self.mu[g, t]
             for g in data.G for t in data.T),
            name='non-negligence_1')

        self.m.addConstrs(
            (sum([self.w[i, t] for i in data.Ig[g]]) <= data.nMax[g, t]*self.y[t-1]
             for g in data.G for t in data.T),
            name='non-negligence_2')

        # Logical constraints
        # ------------------------
        self.m.addConstrs(
            (sum([t*self.e[i, t] for t in data.T]) >= sum([t*self.s[i, t] for t in data.T])
             for i in data.I),
            name='logical_1')

        self.m.addConstrs(
            (sum([self.e[i, t] for t in data.T]) <= 1
             for i in data.I),
            name='logical_2')

        self.m.addConstrs(
            (self.r[i, t] + self.tr[i, t] <= self.u[i, t]
             for i in data.I for t in data.T),
            name='logical_3')

        self.m.addConstrs(
            (sum([self.w[i, t] for t in data.T]) >= self.z[i]
             for i in data.I),
            name='logical_4')

        self.m.update()

        return solution.Solution(
            self.m, dict(s=self.s, tr=self.tr, r=self.r, er=self.er, e=self.e, u=self.u, w=self.w, z=self.z, cr=self.cr, y=self.y, mu=self.mu))
    # --------------------------------------------------------------------------- #
