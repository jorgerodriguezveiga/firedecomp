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
    def __init__(self, problem_data, lambda1, primal=None, option_decomp='G',
        relaxed=False, min_res_penalty=1000000):
        self.primal = primal

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
        self.m = gurobipy.Model("Relaxed_Sual_Problem_LR")

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
        self.md_lambda = self.m.addVars(self.lambda1, vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="lambda")

        self.s  = {}
        self.tr = {}
        self.r  = {}
        self.er = {}
        self.e = {}
        self.u = {}
        self.w = {}
        self.cr = {}
        self.mu = {}
        self.y = {}
        self.z = {}

        for i in self.I:
            for t in self.T:
                self.s[i,t]  = primal.solution.get_variables().get_variable('s')[i,t].X
                self.tr[i,t] = primal.solution.get_variables().get_variable('tr')[i,t].X
                self.r[i,t]  = primal.solution.get_variables().get_variable('r')[i,t].X
                self.er[i,t] = primal.solution.get_variables().get_variable('er')[i,t].X
                self.e[i,t]  = primal.solution.get_variables().get_variable('e')[i,t].X
                self.u[i,t]  = primal.solution.get_variables().get_variable('u')[i,t].getValue()
                self.w[i,t]  = primal.solution.get_variables().get_variable('w')[i,t].getValue()
                self.cr[i,t] = primal.solution.get_variables().get_variable('cr')[i,t].getValue()

        for g in self.G:
            for t in self.T
                self.mu[g,t] = primal.solution..get_variables().get_variable('mu')[g, t].X

        TMU = self.T + [self.min_t-1]
        for t in TMU:
            self.y[t]  = primal.solution..get_variables().get_variable('y')[t].X

        for i in self.I:
            self.z  = primal.solution.get_variables().get_variable('z')[i].getValue()

################################################################################
# PRIVATE METHOD: __sense_opt__()
# OVERWRITE RelaxedPrimalProblem.__sense_opt__()
################################################################################
    def __sense_opt__(self):
        self.__sense_opt__= gurobipy.GRB.MAXIMIZE
