"""Module with wildfire suppression model definition."""

# Python packages
try:
    import pyscipopt as scip
except ModuleNotFoundError:
    pass

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

    def solve(self, solver_options):
        """Solve mathematical model.

        Args:
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
        """
        if solver_options is None:
            solver_options = {'display/verblevel': 0}

        m = self.model
        scip_model = m.model

        # set scip options: https://scip.zib.de/doc/html/PARAMETERS.php
        if isinstance(solver_options, dict):
            for k, v in solver_options.items():
                scip_model.setParam(k, v)


        scip_model.optimize()


        # Todo: check what status number return a solution
        status = scip_model.getStatus()
        if status != "infeasible":
            if self.relaxed is not True:
                # Load variables values
                self.problem_data.resources.update(
                    {i: {'select': sum([scip_model.getVal(m.variables.e[i,t]) for t in self.T])== 1}
                     for i in self.I})

                uval = {(i,t): sum([scip_model.getVal(m.variables.s[i, tind]) for tind in self.T_int.get_names(p_max=t)])
                          - sum([scip_model.getVal(m.variables.e[i, tind]) for tind in self.T_int.get_names(p_max=t-1)])
                        for i in self.I for t in self.T}
                zval = {(i,t): uval[i,t] - scip_model.getVal(m.variables.r[i, t]) - scip_model.getVal(m.variables.tr[i, t])
                        for i in self.I for t in self.T }

                self.problem_data.resources_wildfire.update(
                    {(i, t): {
                        'start': scip_model.getVal(m.variables.s[i, t]) == 1,
                        'travel': scip_model.getVal(m.variables.tr[i, t]) == 1,
                        'rest': scip_model.getVal(m.variables.r[i, t]) == 1,
                        'end_rest': scip_model.getVal(m.variables.er[i, t]) == 1,
                        'end': scip_model.getVal(m.variables.e[i, t]) == 1,
                        'use':  uval[i,t] == 1,
                        'work': zval[i, t] == 1
                    }
                     for i in self.I for t in self.T})

                self.problem_data.groups_wildfire.update(
                    {(g, t): {'num_left_resources': scip_model.getVal(m.variables.mu[g, t])}
                     for g in self.G for t in self.T})

                contained = {t: scip_model.getVal(m.variables.y[t]) == 0 for t in self.T}
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
                config.gurobi.status_info[status]['description'])

        return m
# --------------------------------------------------------------------------- #


# model -----------------------------------------------------------------------
def model(data, relaxed=False):
    """Wildfire suppression model.
    data (:obj:`firedecomp.model.InputModel`): problem data.
    """
    m = scip.Model("wildfire_supression")

    if relaxed is True:
        vtype = "C"
        lb = 0
        ub = 1
    else:
        vtype = "B"
        lb = 0
        ub = 1

    # Variables
    # =========
    # Resources
    # ---------
    s = {}
    tr = {}
    r = {}
    er = {}
    e = {}
    for i in data.I:
        for t in data.T:
            s[i,t] = m.addVar(vtype=vtype, lb=lb, ub=ub, name="start[%s,%s]"%(i,t))
            tr[i,t] = m.addVar(vtype=vtype, lb=lb, ub=ub, name="travel[%s,%s]"%(i,t))
            r[i,t] = m.addVar(vtype=vtype, lb=lb, ub=ub, name="rest[%s,%s]"%(i,t))
            er[i,t] = m.addVar(vtype=vtype, lb=lb, ub=ub, name="end_rest[%s,%s]"%(i,t))
            e[i,t] = m.addVar(vtype=vtype, lb=lb, ub=ub, name="end[%s,%s]"%(i,t))


    # Auxiliar variables
    u = {
        (i, t):
            scip.quicksum(s[i,tind] for tind in data.T_int.get_names(p_max=t))
            - scip.quicksum(e[i,tind] for tind in data.T_int.get_names(p_max=t-1))
        for i in data.I for t in data.T}
    w = {(i, t): u[i, t] - r[i, t] - tr[i, t] for i in data.I for t in data.T}
    z = {i: scip.quicksum(e[i,tind] for tind in data.T) for i in data.I}

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
    y = {t:
            m.addVar(vtype=vtype, lb=lb, ub=ub,name="contention[%s]"%t)
            for t in data.T + [data.min_t-1]}
    mu = {(g,t):
            m.addVar(vtype="C", lb=0,name="missing_resources[%s,%s]"%(g,t))
            for g in data.G for t in data.T}

    # Objective
    # =========
    m.setObjective(sum([data.C[i]*u[i, t] for i in data.I for t in data.T]) +
                   sum([data.P[i] * z[i] for i in data.I]) +
                   sum([data.NVC[t] * y[t-1] for t in data.T]) +
                   sum([data.Mp*mu[g, t] for g in data.G for t in data.T]) +
                   0.001*y[data.max_t]
                   , sense = "minimize")

    # Constraints
    # ===========

    # Wildfire Containment
    # --------------------
    m.addCons(y[data.min_t-1] == 1, name='start_no_contained')

    m.addCons(sum([data.PER[t]*y[t-1] for t in data.T]) <=
                sum([data.PR[i, t]*w[i, t] for i in data.I for t in data.T]),
                name='wildfire_containment_1')

    for t in data.T:
        m.addCons(
            data.M*y[t] >=
             sum([data.PER[t1] for t1 in data.T_int.get_names(p_max=t)])*y[t-1] -
             sum([data.PR[i, t1]*w[i, t1]
                  for i in data.I for t1 in data.T_int.get_names(p_max=t)])
            ,name='wildfire_containment_2[%s]'%t)

    # Start of activity
    # -----------------
    for i in data.I:
        for t in data.T:
            m.addCons(
                data.A[i]*w[i, t] <=
                 sum([tr[i, t1] for t1 in data.T_int.get_names(p_max=t)])
                ,name='start_activity_1[%s,%s]'%(i,t))

    for i in data.I:
        if data.ITW[i] == True:
            m.addCons(
                 s[i, data.min_t] +
                 sum([(data.max_t + 1)*s[i, t]
                      for t in data.T_int.get_names(p_min=data.min_t+1)]) <=
                 data.max_t*z[i]
                ,name='start_activity_2[%s]'%i)

    for i in data.I:
        if data.ITW[i] == False:
            m.addCons(
                sum([s[i, t] for t in data.T]) <= z[i]
                ,name='start_activity_3[%s]'%i)

    # End of Activity
    # ---------------
    for i in data.I:
        for t in data.T:
            m.addCons(
                sum([tr[i, t1] for t1 in data.T_int.get_names(p_min=t-data.TRP[i]+1,
                                                               p_max=t)
                      ]) >= data.TRP[i]*e[i, t]
                ,name='end_activity[%s,%s]'%(i,t))

    # Breaks
    # ------
    for i in data.I:
        for t in data.T:
            m.addCons(
                0 <= cr[i, t]
                ,name='Breaks_1_lb[%s,%s]'%(i,t))

    for i in data.I:
        for t in data.T:
            m.addCons(
                cr[i, t] <= data.WP[i]
                ,name='Breaks_1_ub[%s,%s]'%(i,t))

    for i in data.I:
        for t in data.T:
            m.addCons(
                r[i, t] <= sum([er[i, t1]
                                 for t1 in data.T_int.get_names(p_min=t,
                                                                p_max=t+data.RP[i]-1)])
                ,name='Breaks_2[%s,%s]'%(i,t))

    for i in data.I:
        for t in data.T:
            m.addCons(
                sum([
                    r[i, t1]
                    for t1 in data.T_int.get_names(p_min=t-data.RP[i]+1, p_max=t)]) >=
                 data.RP[i]*er[i, t]
                 if t >= data.min_t - 1 + data.RP[i] else
                 data.CRP[i]*s[i, data.min_t] +
                 sum([r[i, t1] for t1 in data.T_int.get_names(p_max=t)]) >=
                 data.RP[i]*er[i, t]
                ,name='Breaks_3[%s,%s]'%(i,t))

    for i in data.I:
        for t in data.T:
            m.addCons(
                sum([r[i, t1]+tr[i, t1]
                      for t1 in data.T_int.get_names(p_min=t-data.TRP[i],
                                                     p_max=t+data.TRP[i])]) >=
                 len(data.T_int.get_names(p_min=t-data.TRP[i],
                                          p_max=t+data.TRP[i]))*r[i, t]
                ,name='Breaks_4[%s,%s]'%(i,t))

    # Maximum Number of Usage Periods in a Day
    # ----------------------------------------
    for i in data.I:
        m.addCons(
            sum([u[i, t] for t in data.T]) <= data.UP[i] - data.CUP[i]
            ,name='max_usage_periods[%s]'%i)

    # Non-Negligence of Fronts
    # ------------------------
    for g in data.G:
        for t in data.T:
            m.addCons(
                sum([w[i, t] for i in data.Ig[g]]) >= data.nMin[g, t]*y[t-1] - mu[g, t]
                ,name='non-negligence_1[%s,%s]'%(g,t))

    for g in data.G:
        for t in data.T:
            m.addCons(
                sum([w[i, t] for i in data.Ig[g]]) <= data.nMax[g, t]*y[t-1]
                ,name='non-negligence_2[%s,%s]'%(g,t))

    # Logical constraints
    # ------------------------
    for i in data.I:
        m.addCons(
            sum([t*e[i, t] for t in data.T]) >= sum([t*s[i, t] for t in data.T])
            ,name='logical_1[%s]'%i)

    for i in data.I:
        m.addCons(
            sum([e[i, t] for t in data.T]) <= 1
            ,name='logical_2[%s]'%i)

    for i in data.I:
        for t in data.T:
            m.addCons(
                r[i, t] + tr[i, t] <= u[i, t]
                ,name='logical_3[%s,%s]'%(i,t))

    for i in data.I:
        m.addCons(
            sum([w[i, t] for t in data.T]) >= z[i]
            ,name='logical_4[%s]'%i)

    return solution.Solution(
        m, dict(s=s, tr=tr, r=r, er=er, e=e, u=u, w=w, z=z, cr=cr, y=y, mu=mu))
# --------------------------------------------------------------------------- #
