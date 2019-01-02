"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import re


# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Master ----------------------------------------------------------------------
class Master(object):
    def __init__(self, problem_data, min_res_penalty=1000000):
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data
        self.data = Expando()
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self.solution = {}
        self.obj_resources_fix = None
        self.obj_resources_variable = None
        self.obj_wildfire = None
        self.obj_law = None
        self.obj_zeta = None

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

    def __build_variables__(self):
        """Build variables."""
        m = self.model
        data = self.data
        # =========
        # Resources
        # ---------
        s_ub = {(i, t): 1 if (data.ITW[i] is not True) or (t == data.min_t)
                else 0 for i in data.I for t in data.T}
        # cr definition condition
        s_ub.update({
            (i, t): 0 for i in data.I for t in data.T
            if (data.IOW[i] == 1) and
               (t > data.min_t) and
               (t < data.min_t + data.RP[i])})

        self.variables.s = m.addVars(
            data.I, data.T, vtype=gurobipy.GRB.BINARY, lb=0, ub=s_ub,
            name="start")

        self.variables.e = self.model.addVars(
            data.I, data.T, vtype=gurobipy.GRB.BINARY, lb=0, ub=1, name="end")

        self.variables.w = self.model.addVars(
            data.I, data.T, vtype=gurobipy.GRB.BINARY, lb=0, ub=1, name="work")

        # Auxiliar variables
        self.variables.z = {i: self.variables.e.sum(i, '*')
                            for i in data.I}

        self.variables.u = {
            (i, t):
                self.variables.s.sum(i, data.T_int.get_names(p_max=t))
                - self.variables.e.sum(i, data.T_int.get_names(p_max=t - 1))
            for i in data.I for t in data.T}

        self.variables.l = {
            t: sum([data.PR[i, t1]*self.variables.w[i, t1]
                    for i in data.I for t1 in data.T_int.get_names(p_max=t)])
            for t in data.T}

        # Wildfire
        # --------
        self.variables.y = self.model.addVars(
            data.T + [data.min_t - 1], vtype=gurobipy.GRB.BINARY, lb=0, ub=1,
            name="contention")
        self.variables.y[data.min_t - 1].lb = 1

        self.variables.mu = self.model.addVars(
            data.G, data.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0,
            name="missing_resources")

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
        z = self.variables.z
        u = self.variables.u
        y = self.variables.y

        zeta = self.variables.zeta
        mu = self.variables.mu

        self.variables.fix_cost_resources = sum(
            [data.P[i] * z[i] for i in data.I])
        self.variables.variable_cost_resources = sum(
            [data.C[i] * u[i, t] for i in data.I for t in data.T])
        self.variables.wildfire_cost = sum(
            [data.NVC[t] * y[t - 1] for t in data.T])
        self.variables.law_cost = sum(
            [data.Mp * mu[g, t] for g in data.G for t in data.T])
        self.variables.sub_obj = zeta

        self.objective = m.setObjective(
            self.variables.fix_cost_resources +
            self.variables.variable_cost_resources +
            self.variables.wildfire_cost +
            self.variables.law_cost +
            self.variables.sub_obj,
            gurobipy.GRB.MINIMIZE)

    def __build_wildfire_containment_1__(self):
        data = self.data
        y = self.variables.y
        l = self.variables.l

        expr_lhs = \
            sum([data.PER[t] * y[t - 1] for t in data.T])
        expr_rhs = l[data.max_t]

        self.constraints.wildfire_containment_1 = self.model.addConstr(
            expr_lhs <= expr_rhs, name='wildfire_containment_1')

    def __build_wildfire_containment_2__(self):
        data = self.data

        y = self.variables.y
        l = self.variables.l
        PER = data.PER

        expr_lhs = {
            t: sum([PER[t1] for t1 in data.T_int.get_names(p_max=t)])*y[t - 1] -
            l[t]
            for t in data.T}

        expr_rhs = {t: data.M * y[t] for t in data.T}

        self.constraints.wildfire_containment_2 = self.model.addConstrs(
            (expr_lhs[t] <= expr_rhs[t] for t in data.T),
            name='wildfire_containment_2')

    def __build_start_activity_1__(self):
        data = self.data
        w = self.variables.w
        s = self.variables.s
        self.constraints.start_activity_1 = self.model.addConstrs(
            (w[i, t] <=
             sum([s[i, t1] for t1 in data.T_int.get_names(p_max=t-data.A[i])])
             for i in data.I for t in data.T),
            name='start_activity_1')

    def __build_start_activity_2__(self):
        data = self.data
        s = self.variables.s
        z = self.variables.z
        self.constraints.start_activity_2 = self.model.addConstrs(
            (s[i, data.min_t] +
             sum([(data.max_t + 1)*s[i, t]
                  for t in data.T_int.get_names(p_min=data.min_t+1)]) <=
             data.max_t*z[i]
             for i in data.I if data.ITW[i] == True),
            name='start_activity_2')

    def __build_start_activity_3__(self):
        data = self.data
        s = self.variables.s
        z = self.variables.z
        self.constraints.start_activity_3 = self.model.addConstrs(
            (sum([s[i, t] for t in data.T]) <= z[i]
             for i in data.I if data.ITW[i] == False),
            name='start_activity_3')

    def __build_end_activity__(self):
        """Travel time after end working.

        Changed respect to the original paper."""
        data = self.data

        w = self.variables.w
        e = self.variables.e

        expr_lhs = {(i, t): w[i, t] for i in data.I for t in data.T}

        expr_rhs = {
            (i, t): sum([e[i, t1]
                         for t1 in data.T_int.get_names(
                            p_min=t + data.TRP[i])
                         ])
            for i in data.I for t in data.T}

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
        data = self.data

        e = self.variables.e

        expr_lhs = {i: sum([e[i, t] for t in data.T]) for i in data.I}
        expr_rhs = {i: 1 for i in data.I}

        self.constraints.logical_1 = self.model.addConstrs(
            (expr_lhs[i] <= expr_rhs[i]
             for i in data.I),
            name='logical_1')

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
        # Benders constraints
        # -------------------
        self.constraints.opt_int = {}
        self.constraints.opt_start_int = {}
        self.constraints.content_feas_int = {}
        self.constraints.feas_int = {}

        # Wildfire Containment
        # --------------------
        self.__build_wildfire_containment_1__()
        self.__build_wildfire_containment_2__()

        # Start of Activity
        self.__build_start_activity_1__()
        self.__build_start_activity_2__()
        self.__build_start_activity_3__()

        # End of Activity
        # ---------------
        self.__build_end_activity__()

        # Maximum Number of Usage Periods in a Day
        # ----------------------------------------
        self.__build_max_usage_periods__()

        # Non-Negligence of Fronts
        # ------------------------
        self.__build_non_negligence_1__()

        # Logical constraints
        # ------------------------
        self.__build_logical_1__()
        self.__build_logical_2__()
        self.__build_logical_4__()

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

    def add_opt_start_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.opt_start_int)
        self.constraints.opt_start_int[cut_num] = m.addConstr(
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

    def get_obj(self, resources_fix=True, resources_variable=True,
                wildfire=True, law=True, zeta=True):
        """Get objective function value."""
        obj_val = 0
        if resources_fix:
            obj_val += self.obj_resources_fix

        if resources_variable:
            obj_val += self.obj_resources_variable

        if wildfire:
            obj_val += self.obj_wildfire

        if law:
            obj_val += self.obj_law

        if zeta:
            obj_val += self.obj_zeta

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
        variables = self.variables

        # set gurobi options
        if isinstance(solver_options, dict):
            for k, v in solver_options.items():
                m.setParam(k, v)

        m.optimize()

        # Todo: check what status number return a solution
        if m.Status != 3:
            self.obj_resources_fix = self.variables.fix_cost_resources.\
                getValue()
            self.obj_resources_variable = self.variables.\
                variable_cost_resources.getValue()
            self.obj_wildfire = self.variables.wildfire_cost.getValue()
            self.obj_law = self.variables.law_cost.getValue()
            self.obj_zeta = self.variables.sub_obj.x
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

            u = {(i, t): self.variables.u[i, t].getValue() == 1
                 for i in self.data.I for t in self.data.T}
            e = {(i, t): self.variables.e[i, t].x == 1
                 for i in self.data.I for t in self.data.T}
            w = {(i, t): self.variables.w[i, t].x == 1
                 for i in self.data.I for t in self.data.T}
            tr = {(i, t): u[i, t] - w[i, t] == 1
                  for i in self.data.I for t in self.data.T}
            r = {(i, t): False
                 for i in self.data.I for t in self.data.T}

            self.problem_data.resources_wildfire.update(
                {(i, t): {
                    'use': u[i, t],
                    'end': e[i, t],
                    'work': w[i, t],
                    'travel': tr[i, t],
                    'rest': r[i, t]
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
