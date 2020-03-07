"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import re
from io import StringIO
import sys


# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Model -----------------------------------------------------------------------
class Model(object):
    """."""
    def __init__(self, problem_data, relaxed=False, valid_constraints=None):
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")

        if valid_constraints is None:
            valid_constraints = ['contention', 'work', 'max_obj']

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

        self.model_name = 'Model'

        self.valid_constraints = valid_constraints
        self.relaxed = relaxed

        self.data = self.problem_data.data
        self.__build_model__()

    def __build_model__(self):
        self.model = gurobipy.Model(self.model_name)
        self.__build_variables__()
        self.__build_objective__()
        self.__build_constraints__()
        self.model.update()

    def binary_variables_type(self):
        if self.relaxed is True:
            vtype = gurobipy.GRB.CONTINUOUS
            lb = 0
            ub = 1
        else:
            vtype = gurobipy.GRB.BINARY
            lb = 0
            ub = 1
        return {'lb': lb, 'ub': ub, 'vtype': vtype}

    # Variables
    # =========
    # Resources
    # ---------
    def __build_s__(self):
        data = self.data
        var_info = self.binary_variables_type()

        self.variables.s = self.model.addVars(
            data.I, data.T, name="start",
            lb=var_info['lb'], ub=var_info['ub'], vtype=var_info['vtype'])

    def __build_tr__(self):
        data = self.data
        var_info = self.binary_variables_type()

        self.variables.tr = self.model.addVars(
            data.I, data.T, name="travel",
            lb=var_info['lb'], ub=var_info['ub'], vtype=var_info['vtype'])

    def __build_r__(self):
        data = self.data
        var_info = self.binary_variables_type()

        self.variables.r = self.model.addVars(
            data.I, data.T, name="rest",
            lb=var_info['lb'], ub=var_info['ub'], vtype=var_info['vtype'])

    def __build_er__(self):
        data = self.data
        var_info = self.binary_variables_type()

        self.variables.er = self.model.addVars(
            data.I, data.T, name="end_rest",
            lb=var_info['lb'], ub=var_info['ub'], vtype=var_info['vtype'])

    def __build_e__(self):
        data = self.data
        var_info = self.binary_variables_type()

        self.variables.e = self.model.addVars(
            data.I, data.T, name="end",
            lb=var_info['lb'], ub=var_info['ub'], vtype=var_info['vtype'])

        # Auxiliar variables
    def __build_u__(self):
        data = self.data
        s = self.variables.s
        e = self.variables.e

        self.variables.u = {
            (i, t):
                sum(s[i, t] for t in data.T_int(p_max=t))
                - sum(e[i, t] for t in data.T_int(p_max=t-1))
            for i in data.I for t in data.T}

    def __build_w__(self):
        data = self.data
        u = self.variables.u
        r = self.variables.r
        tr = self.variables.tr

        self.variables.w = {
            (i, t): u[i, t] - r[i, t] - tr[i, t]
            for i in data.I for t in data.T
        }

    def __build_z__(self):
        data = self.data
        e = self.variables.e

        self.variables.z = {i: sum(e[i, t] for t in data.T) for i in data.I}

    def __build_cr__(self):
        data = self.data
        s = self.variables.s
        e = self.variables.e
        r = self.variables.r
        er = self.variables.er

        self.variables.cr = {
            (i, t): sum([
                  (t+1-t1)*s[i, t1]
                  - (t-t1)*e[i, t1]
                  - r[i, t1]
                  - data.WP[i]*er[i, t1]
                  for t1 in data.T_int(p_max=t)])
            for i in data.I for t in data.T
            if not data.ITW[i] and not data.IOW[i]}

        self.variables.cr.update({
            (i, t):
                (t+data.CWP[i]-data.CRP[i]) * s[i, data.min_t]
                + sum([
                    (t + 1 - t1 + data.WP[i]) * s[i, t1]
                    for t1 in data.T_int(p_min=data.min_t + 1, p_max=t)])
                - sum([
                    (t - t1) * e[i, t1]
                    + r[i, t1]
                    + data.WP[i] * er[i, t1]
                    for t1 in data.T_int(p_max=t)])
            for i in data.I for t in data.T
            if data.ITW[i] or data.IOW[i]})

        # Wildfire
        # --------
    def __build_y__(self):
        data = self.data
        var_info = self.binary_variables_type()

        self.variables.y = self.model.addVars(
            [data.min_t-1] + data.T, name="contention",
            lb=var_info['lb'], ub=var_info['ub'], vtype=var_info['vtype'])

        self.variables.y[data.min_t - 1].lb = 1

    def __build_mu__(self):
        data = self.data
        self.variables.mu = self.model.addVars(
            data.G, data.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0,
            name="missing_resources")

    def __build_fix_cost_resources__(self):
        data = self.data
        z = self.variables.z

        self.variables.fix_cost_resources = sum(
            [data.P[i] * z[i] for i in data.I])

    def __build_variable_cost_resources__(self):
        data = self.data
        u = self.variables.u

        self.variables.variable_cost_resources = sum(
            [data.C[i] * u[i, t] for i in data.I for t in data.T])

    def __build_wildfire_cost__(self):
        data = self.data
        y = self.variables.y

        self.variables.wildfire_cost = sum(
            [data.NVC[t] * y[t - 1] for t in data.T])

    def __build_law_cost__(self):
        data = self.data
        mu = self.variables.mu

        self.variables.law_cost = sum(
            [data.Mp * mu[g, t] for g in data.G for t in data.T])

    def __build_variables__(self):
        """Build variables."""
        self.__build_s__()
        self.__build_tr__()
        self.__build_r__()
        self.__build_er__()
        self.__build_e__()

        self.__build_u__()
        self.__build_w__()
        self.__build_z__()
        self.__build_cr__()

        self.__build_y__()
        self.__build_mu__()

        self.__build_fix_cost_resources__()
        self.__build_variable_cost_resources__()
        self.__build_wildfire_cost__()
        self.__build_law_cost__()

    def __build_objective__(self):
        """Build objective."""
        m = self.model
        self.objective = m.setObjective(
            self.variables.fix_cost_resources +
            self.variables.variable_cost_resources +
            self.variables.wildfire_cost +
            self.variables.law_cost,
            gurobipy.GRB.MINIMIZE)

    # Valid constraints
    # -----------------
    def __build_valid_constraint_contention__(self):
        """For all t in T:
            y[t] <= y[t-1]
        """
        if 'contention' in self.valid_constraints:
            data = self.data
            y = self.variables.y

            expr_lhs = {t: y[t] for t in data.T}
            expr_rhs = {t: y[t - 1] for t in data.T}

            self.model.addConstrs(
                (expr_lhs[t] <= expr_rhs[t] for t in data.T),
                name='valid_constraint_contention'
            )

    def __build_valid_constraint_work__(self):
        """For all i in I, t in T:
            w[i,t] <= y[t-1]
        """
        if 'work' in self.valid_constraints:
            data = self.data
            w = self.variables.w
            y = self.variables.y

            expr_lhs = {(i, t): w[i, t] for i in data.I for t in data.T}
            expr_rhs = {t: y[t - 1] for t in data.T}

            self.model.addConstrs(
                (expr_lhs[i, t] <= expr_rhs[t] for i in data.I for t in data.T),
                name='valid_constraint_work'
            )

    def __build_wildfire_containment_1__(self):
        data = self.data
        w = self.variables.w
        y = self.variables.y

        wildfire_containment_1_expr = \
            sum([data.PER[t] * y[t - 1] for t in data.T]) <= \
            sum([data.PR[i, t] * w[i, t] for i in data.I for t in data.T])

        self.model.addConstr(
            wildfire_containment_1_expr,
            name='wildfire_containment_1'
        )

    def __build_wildfire_containment_2__(self):
        data = self.data
        w = self.variables.w
        y = self.variables.y

        wildfire_containment_2_expr = (
            data.M*y[t] >=
            sum([data.PER[t1]
                 for t1 in data.T_int(p_max=t)])*y[t-1] -
                 sum([data.PR[i, t1]*w[i, t1]
                      for i in data.I for t1 in data.T_int(p_max=t)])
                 for t in data.T)

        self.model.addConstrs(
            wildfire_containment_2_expr,
            name='wildfire_containment_2'
        )

    # Start of activity
    # -----------------
    def __build_start_activity_1__(self):
        data = self.data
        w = self.variables.w
        tr = self.variables.tr

        self.model.addConstrs(
            (data.A[i]*w[i, t] <=
             sum([tr[i, t1] for t1 in data.T_int(p_max=t)])
             for i in data.I for t in data.T),
            name='start_activity_1')

    def __build_start_activity_2__(self):
        data = self.data
        s = self.variables.s
        z = self.variables.z

        self.model.addConstrs(
            (s[i, data.min_t] +
             sum([(data.max_t + 1)*s[i, t]
                  for t in data.T_int(p_min=data.min_t+1)]) <=
             data.max_t*z[i]
             for i in data.I if data.ITW[i] == True),
            name='start_activity_2')

    def __build_start_activity_3__(self):
        data = self.data
        s = self.variables.s
        z = self.variables.z

        self.model.addConstrs(
            (sum([s[i, t] for t in data.T]) <= z[i]
             for i in data.I if data.ITW[i] == False),
            name='start_activity_3')

    # End of Activity
    # ---------------
    def __build_end_activity__(self):
        data = self.data
        e = self.variables.e
        tr = self.variables.tr

        self.model.addConstrs(
            (sum([tr[i, t1] for t1 in data.T_int(
                    p_min=t-data.TRP[i]+1, p_max=t)
                  ]) >= data.TRP[i]*e[i, t]
             for i in data.I for t in data.T),
            name='end_activity')

    # Breaks
    # ------
    def __build_breaks_1_lb__(self):
        data = self.data
        cr = self.variables.cr

        self.model.addConstrs(
            (0 <= cr[i, t]
             for i in data.I for t in data.T),
            name='breaks_1_lb')

    def __build_breaks_1_ub__(self):
        data = self.data
        cr = self.variables.cr

        self.model.addConstrs(
            (cr[i, t] <= data.WP[i]
             for i in data.I for t in data.T),
            name='breaks_1_ub')

    def __build_breaks_2__(self):
        data = self.data
        r = self.variables.r
        er = self.variables.er

        self.model.addConstrs(
            (r[i, t] <= sum([er[i, t1]
                             for t1 in data.T_int(
                                 p_min=t, p_max=t+data.RP[i]-1)])
             for i in data.I for t in data.T),
            name='breaks_2')

    def __build_breaks_3__(self):
        data = self.data
        r = self.variables.r
        er = self.variables.er
        s = self.variables.s

        self.model.addConstrs(
            (sum([
                r[i, t1]
                for t1 in data.T_int(p_min=t-data.RP[i]+1, p_max=t)]) >=
             data.RP[i]*er[i, t]
             if t >= data.min_t - 1 + data.RP[i] else
             data.CRP[i]*s[i, data.min_t] +
             sum([r[i, t1] for t1 in data.T_int(p_max=t)]) >=
             data.RP[i]*er[i, t]
             for i in data.I for t in data.T),
            name='breaks_3')

    def __build_breaks_4__(self):
        data = self.data
        r = self.variables.r
        tr = self.variables.tr

        self.model.addConstrs(
            (sum([r[i, t1]+tr[i, t1]
                  for t1 in data.T_int(
                      p_min=t-data.TRP[i], p_max=t+data.TRP[i])]) >=
             len(data.T_int(p_min=t-data.TRP[i], p_max=t+data.TRP[i]))*r[i, t]
             for i in data.I for t in data.T),
            name='breaks_4')

    # Maximum Number of Usage Periods in a Day
    # ----------------------------------------
    def __build_max_usage_periods__(self):
        data = self.data
        u = self.variables.u

        self.model.addConstrs(
            (sum([u[i, t] for t in data.T]) <= data.UP[i] - data.CUP[i]
             for i in data.I),
            name='max_usage_periods')

    # Non-Negligence of Fronts
    # ------------------------
    def __build_non_negligence_1__(self):
        data = self.data
        w = self.variables.w
        y = self.variables.y
        mu = self.variables.mu

        self.model.addConstrs(
            (sum([w[i, t] for i in data.Ig[g]]) >=
             data.nMin[g, t]*y[t-1] - mu[g, t]
             for g in data.G for t in data.T),
            name='non-negligence_1')

    def __build_non_negligence_2__(self):
        data = self.data
        w = self.variables.w
        y = self.variables.y

        self.model.addConstrs(
            (sum([w[i, t] for i in data.Ig[g]]) <= data.nMax[g, t]*y[t-1]
             for g in data.G for t in data.T),
            name='non-negligence_2')

    # Logical constraints
    # ------------------------
    def __build_logical_1__(self):
        data = self.data
        e = self.variables.e
        s = self.variables.s

        self.model.addConstrs(
            (sum([t*e[i, t] for t in data.T]) >= sum([t*s[i, t]
             for t in data.T])
             for i in data.I),
            name='logical_1')

    def __build_logical_2__(self):
        data = self.data
        e = self.variables.e

        self.model.addConstrs(
            (sum([e[i, t] for t in data.T]) <= 1
             for i in data.I),
            name='logical_2')

    def __build_logical_3__(self):
        data = self.data
        r = self.variables.r
        tr = self.variables.tr
        u = self.variables.u

        self.model.addConstrs(
            (r[i, t] + tr[i, t] <= u[i, t]
             for i in data.I for t in data.T),
            name='logical_3')

    def __build_logical_4__(self):
        data = self.data
        w = self.variables.w
        z = self.variables.z

        self.model.addConstrs(
            (sum([w[i, t] for t in data.T]) >= z[i]
             for i in data.I),
            name='logical_4')

    def __build_constraints__(self):
        """Build constraints."""
        # Valid Constraints
        # -----------------
        self.__build_valid_constraint_contention__()
        self.__build_valid_constraint_work__()

        # Wildfire Containment
        # --------------------
        self.__build_wildfire_containment_1__()
        self.__build_wildfire_containment_2__()

        # Start of Activity
        # -----------------
        self.__build_start_activity_1__()
        self.__build_start_activity_2__()
        self.__build_start_activity_3__()

        # End of Activity
        # ---------------
        self.__build_end_activity__()

        # Breaks
        # ------
        self.__build_breaks_1_lb__()
        self.__build_breaks_1_ub__()
        self.__build_breaks_2__()
        self.__build_breaks_3__()
        self.__build_breaks_4__()

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

    def solve(self, solver_options):
        """Solve mathematical model.

        Args:
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
        """
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        if solver_options is None:
            solver_options = {'OutputFlag': 0, 'LogToConsole': 0}

        m = self.model
        # set gurobi options
        if isinstance(solver_options, dict):
            for k, v in solver_options.items():
                m.setParam(k, v)

        m.optimize()

        # Reset standard output
        sys.stdout = old_stdout

        # Todo: check what status number return a solution
        if m.Status != 3:
            self.update_problem_with_solution()
        else:
            # log.warning(config.gurobi.status_info[m.Status]['description'])
            pass
        return m.Status

    def update_problem_with_solution(self):
        self.solution = {v.VarName: v.x for v in self.model.getVars()}

        # Load variables values
        if not self.relaxed:
            self.update_objective_function_val()
            self.update_resources()
            self.update_resources_wildfire()
            self.update_groups_wildfire()
            self.update_wildfire()

    def update_objective_function_val(self):
        self.obj_resources_fix = self.get_var_val("fix_cost_resources")
        self.obj_resources_variable = self.get_var_val(
            "variable_cost_resources")
        self.obj_wildfire = self.get_var_val("wildfire_cost")
        self.obj_law = self.get_var_val("law_cost")

    def update_resources(self):
        self.problem_data.resources.update_select(
            {i: round(self.get_var_val("z", i)) == 1
             for i in self.data.I})

    def update_resources_wildfire(self):
        self.problem_data.resources_wildfire.update(
            {(i, t): {
                    'start': round(
                        self.get_var_val("s", i, t)) == 1,
                    'travel': round(
                        self.get_var_val("tr", i, t)) == 1,
                    'rest': round(
                        self.get_var_val("r", i, t)) == 1,
                    'end_rest': round(
                        self.get_var_val("er", i, t)) == 1,
                    'end': round(
                        self.get_var_val("e", i, t)) == 1,
                    'use': round(
                        self.get_var_val("u", i, t)) == 1,
                    'work': round(
                        self.get_var_val("w", i, t)) == 1
                }
             for i in self.data.I for t in self.data.T}
        )

    def update_groups_wildfire(self):
        self.problem_data.groups_wildfire.update(
            {
                (g, t): {'num_left_resources': self.get_var_val("mu", g, t)}
                for g in self.data.G for t in self.data.T
            }
        )

    def update_wildfire(self):
        contained = {
            t: round(self.get_var_val("y", t)) == 0 for t in self.data.T
        }
        contained_period = [t for t, v in contained.items() if v is True]

        if len(contained_period) > 0:
            first_contained = min(contained_period) + 1
        else:
            first_contained = self.max_t + 1

        self.problem_data.wildfire.update(
            {t: {'contained': False if t < first_contained else True}
                for t in self.data.T})

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

    def get_constrvio(self):
        return self.model.constrvio

    def get_var_val(self, name, *index):
        value = 0
        if hasattr(self.variables, name):
            var = getattr(self.variables, name)
            if index:
                if len(index) == 1:
                    index = index[0]
                var = var[index]

            try:
                value = var.x
            except:
                value = var.getValue()

        return value
# --------------------------------------------------------------------------- #
