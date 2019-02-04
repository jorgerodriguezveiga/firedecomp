"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import logging as log

# Package modules
from firedecomp.classes import solution
from firedecomp import config

# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Master ----------------------------------------------------------------------
class RelaxedPrimalProblem(object):

    def __init__(self, problem_data, lambda1, relaxed=False,
        min_res_penalty=1000000):
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")

        self.lambda1 = lambda1
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


# model -----------------------------------------------------------------------
def model(self, relaxed=False):
    """Wildfire suppression model.
    """

    m = gurobipy.Model("wildfire_supression_Relaxed_Primal_Problem_LR")

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
    s = m.addVars(self.I, self.T, vtype=vtype, lb=lb, ub=ub, name="start")
    tr = m.addVars(self.I, self.T, vtype=vtype, lb=lb, ub=ub, name="travel")
    r = m.addVars(self.I, self.T, vtype=vtype, lb=lb, ub=ub, name="rest")
    er = m.addVars(self.I, self.T, vtype=vtype, lb=lb, ub=ub, name="end_rest")
    e = m.addVars(self.I, self.T, vtype=vtype, lb=lb, ub=ub, name="end")

    # Auxiliar variables
    u = {
        (i, t):
            s.sum(i, self.T_int.get_names(p_max=t))
            - e.sum(i, self.T_int.get_names(p_max=t-1))
        for i in self.I for t in self.T}
    w = {(i, t): u[i, t] - r[i, t] - tr[i, t] for i in self.I for t in self.T}
    z = {i: e.sum(i, '*') for i in self.I}

    cr = {(i, t):
          sum([
              (t+1-t1)*s[i, t1]
              - (t-t1)*e[i, t1]
              - r[i, t1]
              - self.WP[i]*er[i, t1]
              for t1 in self.T_int.get_names(p_max=t)])
          for i in self.I for t in self.T
          if not self.ITW[i] and not self.IOW[i]}

    cr.update({
        (i, t):
            (t+self.CWP[i]-self.CRP[i]) * s[i, self.min_t]
            + sum([
                (t + 1 - t1 + self.WP[i]) * s[i, t1]
                for t1 in self.T_int.get_names(p_min=self.min_t+1, p_max=t)])
            - sum([
                (t - t1) * e[i, t1]
                + r[i, t1]
                + self.WP[i] * er[i, t1]
                for t1 in self.T_int.get_names(p_max=t)])
        for i in self.I for t in self.T
        if self.ITW[i] or self.IOW[i]})

    # Wildfire
    # --------
    y = m.addVars(self.T + [self.min_t-1], vtype=vtype, lb=lb, ub=ub,
                  name="contention")
    mu = m.addVars(self.G, self.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0,
                   name="missing_resources")

#########################
    # Wildfire Containment (2) and (3)
    # --------------------
    sum1 = sum([self.PER[t]*y[t-1] for t in self.T])
    sum2 = sum([self.PR[i, t]*w[i, t] for i in self.I for t in self.T])
    Constr1 = sum1 - sum2


    sum1 = sum([self.PER[t1] for t in self.T
        for t1 in self.T_int.get_names(p_max=t)])
    sum2 = sum([self.PR[i, t1]*w[i, t1] for i in self.I for t in self.T
        for t1 in self.T_int.get_names(p_max=t)])
    Constr2 = ( -self.M*y[t] + sum1*y[t-1] - sum2 for t in self.T)

    # Non-Negligence of Fronts (14) and (15)
    # ------------------------
    sum1 = sum([w[i, t] for i in self.Ig[g] for g in self.G for t in self.T])
    sum2 = [self.nMin[g, t]*y[t-1] - mu[g, t] for g in self.G for t in self.T]
    Constr3 = sum2 - sum1

    sum1 = sum([w[i, t] for i in self.Ig[g] for t in self.T])
    sum2 = [self.nMax[g, t]*y[t-1] for g in self.G for t in self.T]
    Constr4 = -sum1 -sum2

#########################

    # Objective
    # =========
    sum1 = sum([self.C[i]*u[i, t] for i in self.I for t in self.T])
    sum2 = sum([self.P[i] * z[i] for i in self.I])
    sum3 = sum([self.NVC[t] * y[t-1] for t in self.T])
    sum4 = sum([self.Mp*mu[g, t] for g in self.G for t in self.T])
    m.setObjective( sum1 + sum2 + sum3 + sum4 +
                   0.001*y[self.max_t] +
                   self.lambda1[1] * Constr1 +
                   self.lambda1[2] * Constr2 +
                   self.lambda1[3] * Constr3 +
                   self.lambda1[4] * Constr4
                   , gurobipy.GRB.MINIMIZE)

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

# model -----------------------------------------------------------------------
def update_lambda1(self, m, lambda1):
    """Wildfire suppression model LR.
    m (:obj:`gurobipy.model`): guroby model.
    data (:obj:`firedecomp.model.InputModel`): problem data.
    lambda1 (:obj:`array<int>`): lambda vector
    """
    self.lambda1 = lambda1
    #m = gurobipy.Model("wildfire_supression_Relaxed_Primal_Problem_LR")

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
    s = m.model.getVarByName("start")
    tr = m.model.getVarByName("travel")
    r = m.model.getVarByName("rest")
    er = m.model.getVarByName("end_rest")
    e = m.model.getVarByName("end")

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
    y = m.model.getVarByName("contention")
    mu = m.model.getVarByName("missing_resources")

#########################
    # Wildfire Containment (2) and (3)
    # --------------------
    sum1 = sum([data.PER[t]*y[t-1] for t in data.T])
    sum2 = sum([data.PR[i, t]*w[i, t] for i in data.I for t in data.T])
    Constr1 = sum1 - sum2

    sum1 = sum([data.PER[t1] for t1 in data.T_int.get_names(p_max=t)
        for t in data.T])
    sum2 = sum([data.PR[i, t1]*w[i, t1] for i in data.I for t1 in
        data.T_int.get_names(p_max=t) for t in data.T])
    Constr2 = ( -data.M*y[t] + sum1*y[t-1] - sum2 for t in data.T)

    # Non-Negligence of Fronts (14) and (15)
    # ------------------------
    Constr3 = (-sum([w[i, t] for i in data.Ig[g]]) +
        data.nMin[g, t]*y[t-1] - mu[g, t] for g in data.G for t in data.T)

    Constr4 = (sum([w[i, t] for i in data.Ig[g]]) - data.nMax[g, t]*y[t-1]
         for g in data.G for t in data.T)
#########################

    # Objective
    # =========
    m.setObjective(sum([data.C[i]*u[i, t] for i in data.I for t in data.T]) +
                   sum([data.P[i] * z[i] for i in data.I]) +
                   sum([data.NVC[t] * y[t-1] for t in data.T]) +
                   sum([data.Mp*mu[g, t] for g in data.G for t in data.T]) +
                   0.001*y[data.max_t] +
                   self.lambda1[1] * (Constr1) +
                   self.lambda1[2] * (Constr2) +
                   self.lambda1[3] * (Constr3) +
                   self.lambda1[4] * (Constr4)
                   , gurobipy.GRB.MINIMIZE)

    m.update()

    return solution.Solution(
        m, dict(s=s, tr=tr, r=r, er=er, e=e, u=u, w=w, z=z, cr=cr, y=y, mu=mu))
# --------------------------------------------------------------------------- #

    def solve(self, solver_options):
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

        return m
