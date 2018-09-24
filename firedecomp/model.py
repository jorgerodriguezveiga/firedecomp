"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy

# Package modules
from firedecomp.classes import solution


class InputModel(object):
    def __init__(self, problem_data, min_res_penalty=1000000):
        if problem_data.period_unit is False:
            raise ValueError("Time unit of the problem is not a period.")

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

        self.PER = problem_data.wildfire.get_info("increment_perimeter")
        self.NVC = problem_data.wildfire.get_info("increment_cost")

        self.EF = problem_data.resources_wildfire.get_info(
            "resources_efficiency")

        self.nMin = problem_data.groups_wildfire.get_info("min_res_groups")
        self.nMax = problem_data.groups_wildfire.get_info("max_res_groups")

        self.PR = {(i, t): self.BPR[i]*self.EF[i, t]
                   for i in self.I for t in self.T}
        self.Mp = min_res_penalty
        self.M = sum([v for k, v in self.PER.items()])
        self.min_t = int(min(self.T))

    def solve(self):
        return model(self)
# --------------------------------------------------------------------------- #


# model -----------------------------------------------------------------------
def model(data):
    """Wildfire suppression model.
    data (:obj:`firedecomp.model.InputModel`): problem data.
    """
    m = gurobipy.Model("wildfire_supression")

    # Variables
    # =========
    # Resources
    # ---------
    s = m.addVars(data.I, data.T, vtype=gurobipy.GRB.BINARY, name="start")
    tr = m.addVars(data.I, data.T, vtype=gurobipy.GRB.BINARY, name="travel")
    r = m.addVars(data.I, data.T, vtype=gurobipy.GRB.BINARY, name="rest")
    er = m.addVars(data.I, data.T, vtype=gurobipy.GRB.BINARY, name="end_rest")
    e = m.addVars(data.I, data.T, vtype=gurobipy.GRB.BINARY, name="end")

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
          if data.ITW[i] == 0 and data.IOW[i] == 0 else
          (t+data.CWP[i]-data.CRP[i]) * s[i, data.min_t]
          + sum([
              (t + 1 - t1 + data.WP[i]) * s[i, t1]
              for t1 in data.T_int.get_names(p_min=data.min_t+1, p_max=t)])
          - sum([
              - (t - t1) * e[i, t1]
              - r[i, t1]
              - data.WP[i] * er[i, t1]
              for t1 in data.T_int.get_names(p_max=t)])
          for i in data.I for t in data.T}

    # Wildfire
    # --------
    y = m.addVars(data.T + [data.min_t-1], vtype=gurobipy.GRB.BINARY,
                  name="contention")
    mu = m.addVars(data.G, data.T, vtype=gurobipy.GRB.CONTINUOUS,
                   name="missing_resources")

    # Objective
    # =========
    m.setObjective(sum([data.C[i]*u[i, t] for i in data.I for t in data.T]) +
                   sum([data.P[i] * z[i] for i in data.I]) +
                   sum([data.NVC[t] * y[t-1] for t in data.T]) +
                   sum([data.Mp*mu[g, t] for g in data.G for t in data.T])
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
            for t1 in data.T_int.get_names(p_min=t, p_max=t-data.RP[i]+1)]) >=
         data.RP[i]*er[i, t]
         if t >= data.RP[i] else
         data.CRP[i]*s[i, data.min_t] +
         sum([r[i, t1] for t1 in data.T_int.get_names(p_max=t)]) >=
         t * er[i, t]
         for i in data.I for t in data.T),
        name='Breaks_3')

    m.addConstrs(
        (sum([r[i, t1]+tr[i, t1]
              for t1 in data.T_int.get_names(p_min=t-data.TRP[i],
                                             p_max=t+data.TRP[i])]) >=
         sum([r[i, t] for t1 in data.T_int.get_names(p_min=t - data.TRP[i],
                                                     p_max=t + data.TRP[i])])
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

    return m
# --------------------------------------------------------------------------- #
