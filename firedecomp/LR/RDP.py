"""Module with lagrangian decomposition methods."""
# Python packages
import gurobipy
import logging as log
import pandas as pd

# Package modules
from firedecomp.classes import solution
from firedecomp import config
from firedecomp.LR import RPP


# Subproblem ------------------------------------------------------------------
class RelaxedDualProblem(RPP.RelaxedPrimalProblem):
    def __init__(self, problem_data, lambda1, primal=None, solution=None,
        option_decomp='G',relaxed=False, min_res_penalty=1000000):
        self.primal = primal
        self.solution = solution
        super().__init__(problem_data, lambda1, relaxed, min_res_penalty)

################################################################################
# PRIVATE METHOD: __get_model__ ()
# OVERWRITE --> model.__get_model__()
################################################################################
    def __get_model__(self, relaxed=False):
        """Create gurobi API model.
            Args:
            relaxed (:bool:): This flag describes if logic values are relaxed.
                              Default ``False``.
            Example: ``{'TimeLimit': 10}``.
        """
        self.__extract_parameters_data_problem__()
        self.__create_gurobi_model__()
        self.__create_vars__()
        self.__create_objfunc__()
        self.m.update()
        model = solution.Solution(
            self.m, dict(mu=self.mu))
        return model

##########################################################################################
# PRIVATE METHOD: __create_gurobi_model__ ()
# OVERWRITE RelaxedPrimalProblem.__create_gurobi_model__()
    def __create_gurobi_model__(self):
        """Create gurobi model.
        """
        self.m = gurobipy.Model("Relaxed_Dual_Problem_LR")

##########################################################################################
# PRIVATE METHOD: __extract_parameters_data_problem__ ()
# OVERWRITE RelaxedPrimalProblem.__extract_parameters_data_problem__()
################################################################################
    def __extract_data_problem__(self):
        """ Extract PARAMETERS fields from data problem
        """
        # SET
        self.__extract_set_data_problem__()
        # PARAMETERS
        self.__extract_parameters_data_problem__()
        # SENSE
        self.__sense_opt__()

##########################################################################################
# PRIVATE METHOD: __create_vars__ ()
# OVERWRITE RelaxedPrimalProblem.__create_vars__()
    def __create_vars__(self):
        print(self.m.getVars())
        self.md_lambda = self.m.addVars(self.lambda1, vtype=gurobipy.GRB.CONTINUOUS,
                                lb=0, ub=1)
        print(self.m.getVars())
        self.s  = {}
        self.tr = {}
        self.r  = {}
        self.er = {}
        self.e = {}
        self.mu = {}
        self.y = {}

        for i in self.I:
            for t in self.T:
                self.s[i,t]  = self.solution.get_model().getVarByName("start["+i+","+str(t)+"]").X
                self.tr[i,t] = self.solution.get_model().getVarByName("travel["+i+","+str(t)+"]").X
                self.r[i,t]  = self.solution.get_model().getVarByName("rest["+i+","+str(t)+"]").X
                self.er[i,t] = self.solution.get_model().getVarByName("end_rest["+i+","+str(t)+"]").X
                self.e[i,t]  = self.solution.get_model().getVarByName("end["+i+","+str(t)+"]").X

        for g in self.G:
            for t in self.T:
                self.mu[g,t] = self.solution.get_model().getVarByName("missing_resources["+g+","+str(t)+"]").X
        self.__create_auxiliar_vars__()
        TMU = self.T + [self.min_t-1]
        for sol_i in self.I:
            for t in TMU:
                self.y[t] = self.solution.get_model().getVarByName("contention["+str(t)+"]").X

################################################################################
# PRIVATE METHOD: __sense_opt__()
# OVERWRITE RelaxedPrimalProblem.__sense_opt__()
################################################################################
    def __sense_opt__(self):
        self.__sense_opt__= gurobipy.GRB.MAXIMIZE

################################################################################
# PRIVATE METHOD: __create_auxiliar_vars__()
# OVERWRITE RelaxedPrimalProblem.__create_auxiliar_vars__()
################################################################################
    def __create_auxiliar_vars__(self):
        self.u = {
            (i, t):
                self.s.sum(i, self.T_int.get_names(p_max=t))
                - self.e.sum(i, self.T_int.get_names(p_max=t-1))
            for i in self.I for t in self.T}

        self.w = {(i, t): self.u[i, t] - self.r[i, t] - self.tr[i, t]
            for i in self.I for t in self.T}

        self.z = {i: self.e.sum(i, '*') for i in self.I}

        self.cr = {(i, t):
              sum([
                  (t+1-t1)*self.s[i, t1]
                  - (t-t1)*self.e[i, t1]
                  - self.r[i, t1]
                  - self.WP[i]*self.er[i, t1]
                  for t1 in self.T_int.get_names(p_max=t)])
              for i in self.I for t in self.T
              if not self.ITW[i] and not self.IOW[i]}

        self.cr.update({
            (i, t):
                (t+self.CWP[i]-self.CRP[i]) * self.s[i, self.min_t]
                + sum([
                    (t + 1 - t1 + self.WP[i]) * self.s[i, t1]
                    for t1 in self.T_int.get_names(p_min=self.min_t+1, p_max=t)])
                - sum([
                    (t - t1) * self.e[i, t1]
                    + self.r[i, t1]
                    + self.WP[i] * self.er[i, t1]
                    for t1 in self.T_int.get_names(p_max=t)])
                for i in self.I for t in self.T
                if self.ITW[i] or self.IOW[i]})

################################################################################
# PRIVATE METHOD: __create_auxiliar_vars__()
# OVERWRITE RelaxedPrimalProblem.__create_auxiliar_vars__()
################################################################################
    def __create_objfunc__(self):

# Wildfire Containment (2) and (3)
        Constr1 = (sum([self.PER[t]*self.y[t-1] for t in self.T])
                    - sum([self.PR[i, t]*self.w[i, t] for i in self.I for t in self.T]))

        Constr2 = sum(self.M*self.y[t] - sum([self.PER[t1] for t1 in
                  self.T_int.get_names(p_max=t)])*self.y[t-1] +
                  sum([self.PR[i, t1]*self.w[i, t1]
                  for i in self.I for t1 in self.T_int.get_names(p_max=t)])
                  for t in self.T)
# Non-Negligence of Fronts (14) and (15)
        Constr3 = sum(-(sum([self.w[i, t]
                    for i in self.Ig[g]])) - (self.nMin[g, t]*self.y[t-1]) + self.mu[g, t]
                    for g in self.G for t in self.T)

        Constr4 = sum(sum([self.w[i, t]
                    for i in self.Ig[g]]) - self.nMax[g, t]*self.y[t-1]
                    for g in self.G for t in self.T)

# Objective
# =========
        sum1 = sum([self.C[i]*self.u[i, t] for i in self.I for t in self.T])
        sum2 = sum([self.P[i]*self.z[i] for i in self.I])
        sum3 = sum([self.NVC[t] * self.y[t-1] for t in self.T])
        sum4 = sum([self.Mp*self.mu[g, t] for g in self.G for t in self.T])

        self.m.setObjective( sum1 + sum2 + sum3 + sum4 +
                       0.001*self.y[self.max_t] +
                       self.lambda1[0] * Constr1 +
                       self.lambda1[1] * Constr2 +
                       self.lambda1[2] * Constr3 +
                       self.lambda1[3] * Constr4,
                       self.sense_opt)
