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
class DecomposedPrimalProblem(RPP.RelaxedPrimalProblem):
    def __init__(self, problem_data, lambda1, resource_i, list_y, relaxed=False,
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
        super().__init__(problem_data, lambda1, relaxed, min_res_penalty)

##########################################################################################
# PRIVATE METHOD: __extract_set_data_problem__ ()
# OVERWRITE RelaxedPrimalProblem.__extract_set_data_problem__()
    def __extract_set_data_problem__(self, relaxed=False):
        """ Extract SET fields from data problem
        """

        self.__extract_set_data_problem_by_resources__(relaxed)

##########################################################################################
# PRIVATE METHOD: __extract_set_data_problem_by_groups__ ()
    def __extract_set_data_problem_by_groups__(self, relaxed=False):
        """ Extract SET fields from data problem
        """
        #  SETS
        self.G = [self.problem_data.get_names("groups")[self.resource_i]]
        self.T = self.problem_data.get_names("wildfire")
        key_group = self.G[0]
        self.I = self.problem_data.groups.get_info('resources')[key_group].get_names()
        dic_group = { key_group : self.problem_data.groups.get_info('resources')[key_group] }
        self.Ig = {
            k: [e.name for e in v]
            for k, v in dic_group.items()}
        self.T_int = self.problem_data.wildfire

##########################################################################################
# PRIVATE METHOD: __extract_set_data_problem_by_resources__ ()
    def __extract_set_data_problem_by_resources__(self, relaxed=False):
        """ Extract SET fields from data problem
        """
        #  SETS
        self.I = [self.problem_data.get_names("resources")[self.resource_i]]
        self.T = self.problem_data.get_names("wildfire")
        key_res = self.I[0];
        key_group = ''
        for group in self.problem_data.groups.elements:
            for res in group.resources:
                if res.name == key_res:
                    key_group = group.name
                    break
        if key_group == '':
            print("ERROR: the model is wrong implemented\n")
        resource_i = self.problem_data.groups.get_info('resources')[key_group].get_element(key_res)
        dic_group = { key_group : [resource_i] }

        self.G = [key_group]

        self.Ig = {
            k: [e.name for e in v]
            for k, v in dic_group.items()}

        self.T_int = self.problem_data.wildfire

##########################################################################################
# PRIVATE METHOD: __create_gurobi_model__ ()
# OVERWRITE RelaxedPrimalProblem.__create_gurobi_model__()
    def __create_gurobi_model__(self):
        """Create gurobi model.
        """
        self.m = gurobipy.Model("Decomposed_Primal_Problem_LR_"+
                                 str(self.resource_i))

################################################################################
# METHOD: return_function_obj()
################################################################################
    def return_function_obj(self, solution):

        self.s = solution.get_variables().get_variable('s')
        self.tr = solution.get_variables().get_variable('tr')
        self.r = solution.get_variables().get_variable('r')
        self.er = solution.get_variables().get_variable('er')
        self.e = solution.get_variables().get_variable('e')

        self.__create_auxiliar_vars__()

        self.y = solution.get_variables().get_variable('y')
        self.mu = solution.get_variables().get_variable('mu')
        self.m.update()
        return self.function_obj.getValue()


################################################################################
# PRIVATE METHOD: __create_var_y__
# OVERWRITE RelaxedPrimalProblem.__create_var_y__()
################################################################################
    def __create_var_y__(self):
        self.sizey = len(self.T + [self.min_t-1])

        self.y = self.m.addVars(self.T + [self.min_t-1], vtype=self.vtype,
                                    lb=self.lb, ub=self.ub, name="contention")

        for i in range(0,self.sizey):
            self.y[i].UB=self.list_y[i]
            self.y[i].LB=self.list_y[i]
            self.y[i].Start=self.list_y[i]

################################################################################
# METHOD: return_LR_obj()
################################################################################
    def return_LR_obj(self, solution):

        self.s = solution.get_variables().get_variable('s')
        self.tr = solution.get_variables().get_variable('tr')
        self.r = solution.get_variables().get_variable('r')
        self.er = solution.get_variables().get_variable('er')
        self.e = solution.get_variables().get_variable('e')

        self.__create_auxiliar_vars__()

        self.y = solution.get_variables().get_variable('y')
        self.mu = solution.get_variables().get_variable('mu')
        self.m.update()
        return self.LR_obj.getValue()

################################################################################
# METHOD: divResources
################################################################################
    def divResources(self):
        self.divResources = 1/len(self.problem_data.get_names('resources'))
