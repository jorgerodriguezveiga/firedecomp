"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import logging as log

# Package modules
from firedecomp.classes import solution
from firedecomp import config


# InputModel ------------------------------------------------------------------
class InputModel(object):
    def __init__(self, problem_data, relaxed=False, min_res_penalty=1000000):
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
        self.model = self.__get_model__()

    def __get_model__(self):
        return model(self, relaxed=self.relaxed)

    def solve(self, solver_options=None):
        """Solve mathematical model.

        Args:
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
        """
        if solver_options is None:
            solver_options = {'OutputFlag': 0}

        m = self.model

        # set gurobi options
        if isinstance(solver_options, dict):
            for k, v in solver_options.items():
                m.model.setParam(k, v)

        m.model.optimize()

        # Todo: check what status number return a solution
        if m.model.Status != 3:
            if self.relaxed is not True:
                # Load variables values
                self.problem_data.resources.update(
                    {i: {'select': m.variables.z[i].getValue() == 1}
                     for i in self.I})

                self.problem_data.resources_wildfire.update(
                    {(i, t): {
                        'start': m.variables.s[i, t].x == 1,
                        'travel': m.variables.tr[i, t].x == 1,
                        'rest': m.variables.r[i, t].x == 1,
                        'end_rest': m.variables.er[i, t].x == 1,
                        'end': m.variables.e[i, t].x == 1,
                        'use': m.variables.u[i, t].getValue() == 1,
                        'work': m.variables.w[i, t].getValue() == 1
                    }
                     for i in self.I for t in self.T})

                self.problem_data.groups_wildfire.update(
                    {(g, t): {'num_left_resources': m.variables.mu[g, t].x}
                     for g in self.G for t in self.T})

                contained = {t: m.variables.y[t].x == 0 for t in self.T}
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
            log.warning(
                config.gurobi.status_info[m.model.Status]['description'])

        return m
# --------------------------------------------------------------------------- #


# model -----------------------------------------------------------------------
def model(data, relaxed=False):
    """Wildfire suppression model.
    data (:obj:`firedecomp.model.InputModel`): problem data.
    """
    m = gurobipy.Model("wildfire_supression")

    if relaxed is True:
        vtype = gurobipy.GRB.CONTINUOUS
        lb = 0
        ub = 1
    else:
        vtype = gurobipy.GRB.BINARY
        lb = 0
        ub = 1

    # Variables
    # =========
    # Resources
    # ---------
    s = m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="start")
    tr = m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="travel")
    r = m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="rest")
    er = m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="end_rest")
    e = m.addVars(data.I, data.T, vtype=vtype, lb=lb, ub=ub, name="end")

    # Auxiliar variables
    u = {
        (i, t):
            s.sum(i, data.T_int.get_names(p_max=t))
            - e.sum(i, data.T_int.get_names(p_max=t-1))
        for i in data.I for t in data.T}
    w = {(i, t): u[i, t] - r[i, t] - tr[i, t] for i in data.I for t in data.T}
    z = {i: e.sum(i, '*') for i in data.I}

    cr = {(i, t):
          sum([
              (t+1-t1)*s[i, t1]
              - (t-t1)*e[i, t1]
              - r[i, t1]
              - data.WP[i]*er[i, t1]
              for t1 in data.T_int.get_names(p_max=t)])
          for i in data.I for t in data.T
          if not data.ITW[i] and not data.IOW[i]}

    cr.update({
        (i, t):
            (t+data.CWP[i]-data.CRP[i]) * s[i, data.min_t]
            + sum([
                (t + 1 - t1 + data.WP[i]) * s[i, t1]
                for t1 in data.T_int.get_names(p_min=data.min_t+1, p_max=t)])
            - sum([
                (t - t1) * e[i, t1]
                + r[i, t1]
                + data.WP[i] * er[i, t1]
                for t1 in data.T_int.get_names(p_max=t)])
        for i in data.I for t in data.T
        if data.ITW[i] or data.IOW[i]})

    # Wildfire
    # --------
    y = m.addVars(data.T + [data.min_t-1], vtype=vtype, lb=lb, ub=ub,
                  name="contention")
    mu = m.addVars(data.G, data.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0,
                   name="missing_resources")

    # Objective
    # =========
    m.setObjective(sum([data.C[i]*u[i, t] for i in data.I for t in data.T]) +
                   sum([data.P[i] * z[i] for i in data.I]) +
                   sum([data.NVC[t] * y[t-1] for t in data.T]) +
                   sum([data.Mp*mu[g, t] for g in data.G for t in data.T]) +
                   0.001*y[data.max_t]
                   , gurobipy.GRB.MINIMIZE)

    # Constraints
    # ===========

    # Wildfire Containment
    # --------------------
    m.addConstr(y[data.min_t-1] == 1, name='start_no_contained')

    m.addConstr(sum([data.PER[t]*y[t-1] for t in data.T]) <=
                sum([data.PR[i, t]*w[i, t] for i in data.I for t in data.T]),
                name='wildfire_containment_1')

    m.addConstrs(
        (data.M*y[t] >=
         sum([data.PER[t1] for t1 in data.T_int.get_names(p_max=t)])*y[t-1] -
         sum([data.PR[i, t1]*w[i, t1]
              for i in data.I for t1 in data.T_int.get_names(p_max=t)])
         for t in data.T),
        name='wildfire_containment_2')

    # Start of activity
    # -----------------
    m.addConstrs(
        (data.A[i]*w[i, t] <=
         sum([tr[i, t1] for t1 in data.T_int.get_names(p_max=t)])
         for i in data.I for t in data.T),
        name='start_activity_1')

    m.addConstrs(
        (s[i, data.min_t] +
         sum([(data.max_t + 1)*s[i, t]
              for t in data.T_int.get_names(p_min=data.min_t+1)]) <=
         data.max_t*z[i]
         for i in data.I if data.ITW[i] == True),
        name='start_activity_2')

    m.addConstrs(
        (sum([s[i, t] for t in data.T]) <= z[i]
         for i in data.I if data.ITW[i] == False),
        name='start_activity_3')

    # End of Activity
    # ---------------
    m.addConstrs(
        (sum([tr[i, t1] for t1 in data.T_int.get_names(p_min=t-data.TRP[i]+1,
                                                       p_max=t)
              ]) >= data.TRP[i]*e[i, t]
         for i in data.I for t in data.T),
        name='end_activity')

    # Breaks
    # ------
    m.addConstrs(
        (0 <= cr[i, t]
         for i in data.I for t in data.T),
        name='Breaks_1_lb')

    m.addConstrs(
        (cr[i, t] <= data.WP[i]
         for i in data.I for t in data.T),
        name='Breaks_1_ub')

    m.addConstrs(
        (r[i, t] <= sum([er[i, t1]
                         for t1 in data.T_int.get_names(p_min=t,
                                                        p_max=t+data.RP[i]-1)])
         for i in data.I for t in data.T),
        name='Breaks_2')

    m.addConstrs(
        (sum([
            r[i, t1]
            for t1 in data.T_int.get_names(p_min=t-data.RP[i]+1, p_max=t)]) >=
         data.RP[i]*er[i, t]
         if t >= data.min_t - 1 + data.RP[i] else
         data.CRP[i]*s[i, data.min_t] +
         sum([r[i, t1] for t1 in data.T_int.get_names(p_max=t)]) >=
         data.RP[i]*er[i, t]
         for i in data.I for t in data.T),
        name='Breaks_3')

    m.addConstrs(
        (sum([r[i, t1]+tr[i, t1]
              for t1 in data.T_int.get_names(p_min=t-data.TRP[i],
                                             p_max=t+data.TRP[i])]) >=
         len(data.T_int.get_names(p_min=t-data.TRP[i],
                                  p_max=t+data.TRP[i]))*r[i, t]
         for i in data.I for t in data.T),
        name='Breaks_4')

    # Maximum Number of Usage Periods in a Day
    # ----------------------------------------
    m.addConstrs(
        (sum([u[i, t] for t in data.T]) <= data.UP[i] - data.CUP[i]
         for i in data.I),
        name='max_usage_periods')

    # Non-Negligence of Fronts
    # ------------------------
    m.addConstrs(
        (sum([w[i, t] for i in data.Ig[g]]) >= data.nMin[g, t]*y[t-1] - mu[g, t]
         for g in data.G for t in data.T),
        name='non-negligence_1')

    m.addConstrs(
        (sum([w[i, t] for i in data.Ig[g]]) <= data.nMax[g, t]*y[t-1]
         for g in data.G for t in data.T),
        name='non-negligence_2')

    # Logical constraints
    # ------------------------
    m.addConstrs(
        (sum([t*e[i, t] for t in data.T]) >= sum([t*s[i, t] for t in data.T])
         for i in data.I),
        name='logical_1')

    m.addConstrs(
        (sum([e[i, t] for t in data.T]) <= 1
         for i in data.I),
        name='logical_2')

    m.addConstrs(
        (r[i, t] + tr[i, t] <= u[i, t]
         for i in data.I for t in data.T),
        name='logical_3')

    m.addConstrs(
        (sum([w[i, t] for t in data.T]) >= z[i]
         for i in data.I),
        name='logical_4')

    m.update()

    return solution.Solution(
        m, dict(s=s, tr=tr, r=r, er=er, e=e, u=u, w=w, z=z, cr=cr, y=y, mu=mu))
# --------------------------------------------------------------------------- #





################################################################################
# METHOD: __set_solution_in_RPP__
################################################################################
    def insert_soludion(self, decomp_solutions, option_decomp, lambda1,
                                                solver_options, groups=None):

        # extract lambdas
        self.lambdas1 = lambda1
        # extract solutions
        if (option_decomp == 'R'):
            counter = 0
            for sol_i in self.I:
                for t in self.T:
                    # fix variables
                    self.init_vars(self.s, 's', decomp_solutions[counter], sol_i, t)
                    self.init_vars(self.tr, 'tr', decomp_solutions[counter], sol_i, t)
                    self.init_vars(self.r, 'r', decomp_solutions[counter], sol_i, t)
                    self.init_vars(self.er, 'er', decomp_solutions[counter], sol_i, t)
                    self.init_vars(self.e, 'e', decomp_solutions[counter], sol_i, t)
                    self.init_vars(self.mu, 'mu', decomp_solutions[counter], groups[counter], t)
                counter = counter + 1
            self.__create_auxiliar_vars__()
            counter = 0
            TMU = self.T + [self.min_t-1]
            for sol_i in self.I:
                for t in TMU:
                    self.init_vars_v(self.y, 'y', decomp_solutions[counter], t)

        elif (option_decomp == 'G'):
            for solution_i in decomp_solutions:
                for t in self.T:
                    self.s[i,t]  = decomp_solutions[0].get_variables().get_variable('s')[i,t].X
        else:
            print("Error[0] Use 'G' to divide by groups, or 'R' to divide by resources ")
            sys.exit()
        self.__create_objfunc__()
        self.__create_constraints__()
        self.m.update()
        self.model = solution.Solution(
            self.m, dict(s=self.s, tr=self.tr, r=self.r, er=self.er, e=self.e, u=self.u,
            w=self.w, z=self.z, cr=self.cr, y=self.y, mu=self.mu))

################################################################################
# METHOD: fix_vars
################################################################################
    def init_vars(self, var, varstr, solution, sol_i, t):
        value = solution.get_variables().get_variable(varstr)[sol_i,t].X
        var[sol_i,t].start=value

################################################################################
# METHOD: fix_vars
################################################################################
    def init_vars_v(self, var, varstr, solution, t):
        value = solution.get_variables().get_variable(varstr)[t].X
        var[t].start=value
