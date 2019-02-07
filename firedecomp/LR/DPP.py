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
    def __init__(self, problem_data, lambda1, resource_i, relaxed=False,
        min_res_penalty=1000000):
        self.resource_i= resource_i
        super().__init__(problem_data, lambda1, relaxed, min_res_penalty)

##########################################################################################
# PRIVATE METHOD: __extract_set_data_problem__ ()
# OVERWRITE RelaxedPrimalProblem.__extract_set_data_problem__()
    def __extract_set_data_problem__(self, relaxed=False):
        """ Extract SET fields from data problem
        """
        #  SETS
#        self.I = [self.problem_data.get_names("resources")[self.resource_i]]
        self.G = [self.problem_data.get_names("groups")[self.resource_i]]
        self.T = self.problem_data.get_names("wildfire")

        key_group = self.G[0]
        self.I = self.problem_data.groups.get_info('resources')[key_group].get_names()
        dic_group = { key_group : self.problem_data.groups.get_info('resources')[key_group] }

        print(dic_group.items())

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

##########################################################################################
# BORRADOR DESCOMPOSICION POR INDIVIDUOS
# PRIVATE METHOD: __extract_set_data_problem__ ()
# OVERWRITE RelaxedPrimalProblem.__extract_set_data_problem__()
    def __extract_set_data_problem_OLD__(self, relaxed=False):
        """ Extract SET fields from data problem
        """
        #  SETS
        self.I = [self.problem_data.get_names("resources")[self.resource_i]]
        #self.G = self.problem_data.get_names("groups")
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

#
#        dic_group = self.problem_data.groups.get_info('resources')
#        for k in self.problem_data.groups.get_info('resources').keys():
#            if k != key_group:
#                del dic_group[k]
#        print(dic_group.items())
#
#
        self.Ig = {
            k: [e.name for e in v]
            for k, v in dic_group.items()}

        self.T_int = self.problem_data.wildfire
