"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import re
import logging as log


# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Master ----------------------------------------------------------------------
class Master(object):
    def __init__(self, problem_data, valid_constraints=None):
        """Build Master problem.

        Args:
            problem_data: problem.
            valid_constraints: list with desired valid constraints. Options
                allowed: 'contention', 'work', 'max_obj'. If None all are
                considered. Defaults to None.
        """
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")

        if valid_constraints is None:
            valid_constraints = ['contention', 'work', 'max_obj']

        self.problem_data = problem_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self.solution = {}
        self.obj_resources_fix = None
        self.obj_resources_variable = None
        self.obj_wildfire = None
        self.obj_law = None
        self.max_obj = None
        self.valid_constraints = valid_constraints

        self.__start_info__ = {}
        self.__build_model__()

    def update_model(self):
        self.__build_model__()

    def __build_model__(self):
        self.__build_data__()
        self.model = gurobipy.Model("Master")
        self.__build_variables__()
        self.__build_objective__()
        self.__build_constraints__()
        self.model.update()

    def __build_data__(self):
        self.data = self.problem_data.period_data

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
                self.variables.s.sum(i, data.T_int(p_max=t))
                - self.variables.e.sum(i, data.T_int(p_max=t - 1))
            for i in data.I for t in data.T}

        self.variables.l = {
            t: sum([data.PR[i, t1]*self.variables.w[i, t1]
                    for i in data.I for t1 in data.T_int(p_max=t)])
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

    def use_warm_start(self):
        res_wild = self.problem_data.resources_wildfire
        wild = self.problem_data.wildfire
        grp_wild = self.problem_data.groups_wildfire

        for t in self.data.T:
            if isinstance(wild[t].contained, (bool, int, float)):
                self.variables.y[t].start = int(not wild[t].contained)
            for i in self.data.I:
                if isinstance(res_wild[i, t].start, (bool, int, float)):
                    self.variables.s[i, t].start = int(res_wild[i, t].start)
                if isinstance(res_wild[i, t].end, (bool, int, float)):
                    self.variables.e[i, t].start = int(res_wild[i, t].end)
                if isinstance(res_wild[i, t].work, (bool, int, float)):
                    self.variables.w[i, t].start = int(res_wild[i, t].work)
            for g in self.data.G:
                if isinstance(grp_wild[g, t].num_left_resources, (int, float)):
                    self.variables.mu[g, t].start = \
                        grp_wild[g, t].num_left_resources

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

        mu = self.variables.mu

        self.variables.fix_cost_resources = sum(
            [data.P[i] * z[i] for i in data.I])
        self.variables.variable_cost_resources = sum(
            [data.C[i] * u[i, t] for i in data.I for t in data.T])
        self.variables.wildfire_cost = sum(
            [data.NVC[t] * y[t - 1] for t in data.T])
        self.variables.law_cost = sum(
            [data.Mp * mu[g, t] for g in data.G for t in data.T])

        self.objective = m.setObjective(
            self.variables.fix_cost_resources +
            self.variables.variable_cost_resources +
            self.variables.wildfire_cost +
            self.variables.law_cost,
            gurobipy.GRB.MINIMIZE)

    def __build_valid_constraint_contention__(self):
        y = self.variables.y

        expr_lhs = {t: y[t] for t in self.data.T}
        expr_rhs = {t: y[t - 1] for t in self.data.T}

        self.constraints.wildfire_containment_1 = self.model.addConstrs(
            (expr_lhs[t] <= expr_rhs[t] for t in self.data.T),
            name='valid_constraint_contention')

    def __build_valid_constraint_work__(self):
        """w_{it} <= y_{t - 1}"""
        y = self.variables.y
        w = self.variables.w

        expr_lhs = {(i, t): w[i, t] for i in self.data.I for t in self.data.T}
        expr_rhs = {t: y[t - 1] for t in self.data.T}

        self.constraints.wildfire_containment_1 = self.model.addConstrs(
            (expr_lhs[i, t] <= expr_rhs[t]
             for i in self.data.I for t in self.data.T),
            name='valid_constraint_contention')

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
            t: sum([PER[t1] for t1 in data.T_int(p_max=t)])*y[t - 1] - l[t]
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
             sum([s[i, t1] for t1 in data.T_int(p_max=t-data.A[i])])
             for i in data.I for t in data.T),
            name='start_activity_1')

    def __build_start_activity_2__(self):
        data = self.data
        s = self.variables.s
        z = self.variables.z
        self.constraints.start_activity_2 = self.model.addConstrs(
            (s[i, data.min_t] +
             sum([(data.max_t + 1)*s[i, t]
                  for t in data.T_int(p_min=data.min_t+1)]) <= data.max_t*z[i]
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
            (i, t): sum([e[i, t1] for t1 in data.T_int(p_min=t + data.TRP[i])])
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
        self.constraints.opt_start_int = {}
        self.constraints.content_feas_int = {}
        self.constraints.feas_int = {}
        self.constraints.max_obj = None

        # Start info
        # ----------
        self.__build_opt_start_init_cuts__()

        # Max obj
        # -------
        if 'max_obj' in self.valid_constraints:
            log.info("Add max_obj valid constraint.")
            self.__build_max_obj_cut__()

        # Valid constraints
        # -----------------
        if 'contention' in self.valid_constraints:
            log.info("Add contention valid constraint.")
            self.__build_valid_constraint_contention__()

        if 'work' in self.valid_constraints:
            log.info("Add work valid constraint.")
            self.__build_valid_constraint_work__()

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
        self.__build_non_negligence_2__()

        # Logical constraints
        # ------------------------
        self.__build_logical_1__()
        self.__build_logical_2__()
        self.__build_logical_4__()

    def __build_max_obj_cut__(self):
        if self.max_obj is not None:
            m = self.model
            data = self.data
            z = self.variables.z
            u = self.variables.u
            y = self.variables.y

            mu = self.variables.mu

            self.variables.fix_cost_resources = sum(
                [data.P[i] * z[i] for i in data.I])
            self.variables.variable_cost_resources = sum(
                [data.C[i] * u[i, t] for i in data.I for t in data.T])
            self.variables.wildfire_cost = sum(
                [data.NVC[t] * y[t - 1] for t in data.T])
            self.variables.law_cost = sum(
                [data.Mp * mu[g, t] for g in data.G for t in data.T])

            lhs = self.max_obj
            rhs = \
                self.variables.fix_cost_resources + \
                self.variables.variable_cost_resources + \
                self.variables.wildfire_cost + \
                self.variables.law_cost

            self.constraints.max_obj = m.addConstr(lhs >= rhs, name='max_obj')
            m.update()

    def __build_opt_start_init_cuts__(self):
        for it, v in self.__start_info__.items():
            work = v['work']
            max_t = int(max(self.problem_data.get_names("wildfire")))
            coeffs = {'start[{},{}]'.format(it[0], it[1]): - max_t}
            coeffs.update({"work[{},{}]".format(k[0], k[1]): - 1
                           for k, v in work.items() if v == 0})
            rhs = - max_t
            self.add_opt_start_int_cut(coeffs, rhs)

    def add_opt_start_int_cut(self, vars_coeffs, rhs):
        m = self.model
        cut_num = len(self.constraints.opt_start_int)
        lhs = sum(
            coeff*m.getVarByName(var) for var, coeff in vars_coeffs.items()
            if m.getVarByName(var) is not None)
        self.constraints.opt_start_int[cut_num] = \
            m.addConstr(lhs >= rhs, name='start_opt_int[{}]'.format(cut_num))
        m.update()

    def add_feas_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.feas_int)
        self.constraints.feas_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items()
                if m.getVarByName(var) is not None) >= rhs,
            name='feas_int[{}]'.format(cut_num)
        )
        m.update()

    def add_contention_feas_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.content_feas_int)
        self.constraints.content_feas_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items()
                if m.getVarByName(var) is not None) >= rhs,
            name='content_feas_int[{}]'.format(cut_num)
        )
        m.update()

    def get_obj(self, resources_fix=True, resources_variable=True,
                wildfire=True, law=True):
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

        # Check if exist a solution
        if m.SolCount >= 1 and m.Status != 3:
            orig_data = self.problem_data.data
            self.obj_resources_fix = self.variables.fix_cost_resources.\
                getValue()
            self.obj_resources_variable = self.variables.\
                variable_cost_resources.getValue()
            self.obj_wildfire = self.variables.wildfire_cost.getValue()
            self.obj_law = self.variables.law_cost.getValue()
            self.solution = {v.VarName: v.x for v in self.model.getVars()}

            # Load variables values
            self.problem_data.resources.update(
                {i: {'select': round(variables.z[i].getValue()) == 1}
                 for i in self.data.I})

            s = {(i, t): round(variables.s[i, t].x) == 1
                 if t <= self.data.max_t else False
                 for i in self.data.I for t in orig_data.T}
            u = {(i, t): round(self.variables.u[i, t].getValue()) == 1
                 if t <= self.data.max_t else False
                 for i in self.data.I for t in orig_data.T}
            e = {(i, t): round(self.variables.e[i, t].x) == 1
                 if t <= self.data.max_t else False
                 for i in self.data.I for t in orig_data.T}
            w = {(i, t): round(self.variables.w[i, t].x) == 1
                 if t <= self.data.max_t else False
                 for i in self.data.I for t in orig_data.T}
            start = {k[0]: k[1] for k, v in s.items() if v is True}
            r = {(i, t): False for i in self.data.I for t in orig_data.T}
            r.update({
                (i, t): round(
                    self.__start_info__[i, start[i]]['rest'][i, t]*u[i, t])
                == 1
                if (i, start[i]) in self.__start_info__ else
                False
                for i in self.data.I for t in orig_data.T if i in start})

            er = {
                (i, t): True
                if (r[i, t] is True) and (r[i, t + 1] is not True)
                else False
                for i in self.data.I for t in orig_data.T_int(
                    p_max=orig_data.max_t - 1)
            }
            er.update({
                (i, orig_data.max_t): True
                if r[i, orig_data.max_t] is True else False
                for i in self.data.I})
            tr = {
                 (i, t): u[i, t] - w[i, t] - r[i, t] == 1
                 for i in self.data.I for t in orig_data.T
            }

            self.problem_data.resources_wildfire.update(
                {(i, t): {
                    'start': s[i, t],
                    'use': u[i, t],
                    'end': e[i, t],
                    'work': w[i, t],
                    'travel': tr[i, t],
                    'rest': r[i, t],
                    'end_rest': er[i, t]
                }
                 for i in self.data.I for t in orig_data.T})

            self.problem_data.groups_wildfire.update(
                {(g, t): {'num_left_resources': self.variables.mu[g, t].x}
                 if t <= self.data.max_t else {'num_left_resources': 0}
                 for g in self.data.G for t in orig_data.T})

            contained = {t: self.variables.y[t].x == 0
                         if t <= self.data.max_t else True
                         for t in orig_data.T}
            contained_period = [t for t, v in contained.items()
                                if v is True]

            if len(contained_period) > 0:
                first_contained = min(contained_period) + 1
            else:
                first_contained = orig_data.max_t + 1

            self.problem_data.wildfire.update(
                {t: {'contained': False if t < first_contained else True}
                 for t in orig_data.T})
        else:
            pass
        return m.Status
# --------------------------------------------------------------------------- #
