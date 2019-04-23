"""Module with lagrangian decomposition methods."""
# Python packages
import gurobipy
import logging as log
import pandas as pd
import copy

# Package modules
from firedecomp.classes import solution
from firedecomp import config
from firedecomp.LR import RPP


# Subproblem ------------------------------------------------------------------
class DecomposedPrimalProblem(RPP.RelaxedPrimalProblem):
    def __init__(self, problem_data, lambda1, resource_i, list_y, sol, relaxed=False,
                 min_res_penalty=1000000):
        """Initialize the DecomposedPrimalProblem.

        Args:
            problem_data (:obj:`Problem`): problem data.
            lambda1  (:obj:`list`): list of numeric values of lambdas (integers)
            resource_i (:obj:`int`): index resource
            option_decomp (:obj:`str`): maximum number of iterations. Defaults to
                Defaults to 0.01.
            relaxed (:obj:`float`):  Defaults to 0.01.
            min_res_penalty (:obj:`int`): .
                Default to 1000000
        """
        self.resource_i= resource_i
        self.list_y = list_y
        self.solution = sol
        super().__init__(problem_data, lambda1, relaxed, min_res_penalty)


##########################################################################################
# PRIVATE METHOD: __extract_set_data_problem__ ()
# OVERWRITE RelaxedPrimalProblem.__extract_set_data_problem__()
##########################################################################################
#    def __extract_set_data_problem__(self, relaxed=False):
#        """ Extract SET fields from data problem
#        """
#
#        self.__extract_set_data_problem_by_resources__(relaxed)

##########################################################################################
# PRIVATE METHOD: __create_gurobi_model__ ()
# OVERWRITE RelaxedPrimalProblem.__create_gurobi_model__()
    def __create_gurobi_model__(self):
        """Create gurobi model.
        """
        self.m = gurobipy.Model("Decomposed_Primal_Problem_LR_"+
                                 str(self.resource_i))

################################################################################
# PRIVATE METHOD: __create_var_y__
# OVERWRITE RelaxedPrimalProblem.__create_var_y__()
################################################################################
    def __create_var_y_and_fixed_vars__(self):
        self.sizey = len(self.T + [self.min_t-1])
        self.y = self.m.addVars(self.T + [self.min_t-1], vtype=self.vtype,
                                    lb=self.lb, ub=self.ub, name="contention")
        # fixed y
        for i in range(0,self.sizey):
            self.y[i].UB=self.list_y[i]
            self.y[i].LB=self.list_y[i]
            self.y[i].Start=self.list_y[i]
        ##########################################
        ##########################################
        # fixed solution vars
        for i in range(0,len(self.I)):
            if (i != self.resource_i):
                for t in self.T:
                    ind = self.I[i]
                    # VAR S
                    #print(self.s[ind,t].UB)
                    #print(self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").start)
                    self.s[ind,t].UB     = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").start
                    self.s[ind,t].LB     = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").start
                    self.s[ind,t].Start  = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").start
                    # VAR TR
                    self.tr[ind,t].UB     = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").start
                    self.tr[ind,t].LB     = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").start
                    self.tr[ind,t].Start  = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").start
                    # VAR R
                    self.r[ind,t].UB     = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").start
                    self.r[ind,t].LB     = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").start
                    self.r[ind,t].Start  = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").start
                    # VAR ER
                    self.er[ind,t].UB     = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").start
                    self.er[ind,t].LB     = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").start
                    self.er[ind,t].Start  = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").start
                    # VAR E
                    self.e[ind,t].UB     = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").start
                    self.e[ind,t].LB     = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").start
                    self.e[ind,t].Start  = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").start
        # fixed solution vars
        #for g in self.G:
        #    if (g != self.resource_g ):
        #        for t in self.T:
        #            # VAR MU
        #            self.mu[g,t].UB     = self.solution.get_variables().mu[g,t].getValue()
        #            self.mu[g,t].LB     = self.solution.get_variables().mu[g,t].getValue()
        #            self.mu[g,t].Start  = self.solution.get_variables().mu[g,t].getValue()

##########################################################################################
# PRIVATE METHOD: __extract_set_data_problem_by_resources__ ()
##########################################################################################
#    def __extract_set_data_problem_by_resources__(self, relaxed=False):
#        """ Extract SET fields from data problem
#        """
#        #  SETS
#        self.I = self.problem_data.get_names("resources")
#        self.RI = [self.problem_data.get_names("resources")[self.resource_i]]
#        self.T = self.problem_data.get_names("wildfire")
#        self.G = self.problem_data.get_names("groups")
#        key_res = self.RI[0]
#        key_group = ''
#        for group in self.problem_data.groups.elements:
#            for res in group.resources:
#                if res.name == key_res:
#                    key_group = group.name
#                    break
#        if key_group == '':
#            print("ERROR: the model is wrong implemented\n")
#        resource_i = self.problem_data.groups.get_info('resources')[key_group].get_element(key_res)
#        dic_group = { key_group : [resource_i] }
#        self.resource_g = [key_group]
#        self.Ig = {
#            k: [e.name for e in v]
#            for k, v in dic_group.items()}
#        self.T_int = self.problem_data.wildfire

################################################################################
# METHOD: UPDATE LAMBDA1
################################################################################
    def update_lambda1(self, lambda1, sol):
        """Update lambda in DPP model
            Args:
            lambda1 (:obj:`list`): Array with lambda values.
        """

        self.solution = sol
        self.lambda1 = lambda1
        self.__create_vars__()
        self.__create_objfunc__()
        self.__create_constraints__()
        self.m.update()
        self.model = solution.Solution(
            self.m, dict(s=self.s, tr=self.tr, r=self.r, er=self.er, e=self.e, u=self.u,
            w=self.w, z=self.z, cr=self.cr, y=self.y, mu=self.mu))



##########################################################################################
# PRIVATE METHOD: __extract_set_data_problem_by_resources__ ()
##########################################################################################
#    def __extract_set_data_problem_by_groups__(self, relaxed=False):
#        """ Extract SET fields from data problem
#        """
#        #  SETS
#        self.G = [self.problem_data.get_names("groups")[self.group_i]]
#        self.T = self.problem_data.get_names("wildfire")
#        self.I = []
#        for res in self.problem_data.groups[self.G[0]].resources:
#            self.I.append(res.name)
#        dic_group = { self.G[0] : self.I }

#        self.Ig = {
#            k: [e for e in v]
#            for k, v in dic_group.items()}
#        self.T_int = self.problem_data.wildfire
