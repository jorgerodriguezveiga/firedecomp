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
    def __init__(self, problem_data, lambda1, resource_i, list_y, sol, NL, relaxed=False,
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
        self.index_L = []
        self.NL = NL
        for i in range(0,NL):
            self.index_L.append(0.0)


        super().__init__(problem_data, lambda1, relaxed, min_res_penalty)

    def return_index_L(self):
        return self.index_L
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
            for t in self.T:
                ind = self.I[i]
                # VAR S
                if (i != self.resource_i):
                    self.s[ind,t].UB     = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").start
                    self.s[ind,t].LB     = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").start
                else:
                    self.s[ind,t].UB     = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").UB
                    self.s[ind,t].LB     = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").LB
                self.s[ind,t].Start  = self.solution.get_model().getVarByName("start["+ind+","+str(t)+"]").start
                # VAR TR
                if (i != self.resource_i):
                    self.tr[ind,t].UB     = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").start
                    self.tr[ind,t].LB     = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").start
                else:
                    self.tr[ind,t].UB     = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").UB
                    self.tr[ind,t].LB     = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").LB
                self.tr[ind,t].Start  = self.solution.get_model().getVarByName("travel["+ind+","+str(t)+"]").start
                # VAR R
                if (i != self.resource_i):
                    self.r[ind,t].UB     = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").start
                    self.r[ind,t].LB     = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").start
                else:
                    self.r[ind,t].UB     = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").UB
                    self.r[ind,t].LB     = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").LB
                self.r[ind,t].Start  = self.solution.get_model().getVarByName("rest["+ind+","+str(t)+"]").start
                # VAR ER
                if (i != self.resource_i):
                    self.er[ind,t].UB     = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").start
                    self.er[ind,t].LB     = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").start
                else:
                    self.er[ind,t].UB     = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").UB
                    self.er[ind,t].LB     = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").LB
                self.er[ind,t].Start  = self.solution.get_model().getVarByName("end_rest["+ind+","+str(t)+"]").start
                # VAR E
                if (i != self.resource_i):
                    self.e[ind,t].UB     = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").start
                    self.e[ind,t].LB     = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").start
                else:
                    self.e[ind,t].UB     = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").UB
                    self.e[ind,t].LB     = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").LB
                self.e[ind,t].Start  = self.solution.get_model().getVarByName("end["+ind+","+str(t)+"]").start
        # fixed solution vars
        for gro in self.G:
                for tt in self.T:
                    # VAR MU
                    self.mu[gro,tt].Start  = self.solution.get_model().getVarByName("missing_resources["+gro+","+str(tt)+"]").start


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


################################################################################
# PRIVATE METHOD: __create_objfunc__()
################################################################################
    def __create_objfunc__(self):
# Wildfire Containment (2) and (3)
        Constr1 = (sum([self.PER[t]*self.y[t-1] for t in self.T]) -
                    sum([self.PR[i, t]*self.w[i, t]
                     for i in self.I for t in self.T]))


        list_Constr2 = list(-1.0*self.M*self.y[t] + sum([self.PER[t1] for t1 in self.T_int.get_names(p_max=t)])*self.y[t-1]
                    - sum([self.PR[i, t1]*self.w[i, t1] for i in self.I for t1 in self.T_int.get_names(p_max=t)])
                    for t in self.T)


# Non-Negligence of Fronts (14) and (15)
        list_Constr3 = list((-1.0*sum([self.w[i, t] for i in self.Ig[g]])) - (self.nMin[g, t]*self.y[t-1] + self.mu[g, t])
                    for g in self.G for t in self.T)
        list_Constr4 = list(sum([self.w[i, t] for i in self.Ig[g]]) - self.nMax[g, t]*self.y[t-1]
                    for g in self.G for t in self.T)

# Objective
# =========
        self.function_obj = (sum([self.C[i]*self.u[i, t] for i in self.I for t in self.T]) +
                       sum([self.P[i] * self.z[i] for i in self.I]) +
                       sum([self.NVC[t] * self.y[t-1] for t in self.T]) +
                       sum([self.Mp*self.mu[g, t] for g in self.G for t in self.T]) +
                       0.001*self.y[self.max_t])
# CERO
        self.index_L[0] =  1
        self.lambda1[0] = self.lambda1[0]
        self.LR_obj = self.lambda1[0] * Constr1
        self.LR_obj_const = []
        self.LR_obj_const.append(Constr1)
# UNO
        anula=1
#        if self.resource_i != 0 :
#            anula=0
        counter=1
        for i in range(counter,counter+len(list_Constr2)):
            if anula == 1:
                self.index_L[i] = 1
            self.lambda1[i] = self.lambda1[i] * anula
            self.LR_obj = self.LR_obj + self.lambda1[i] * list_Constr2[i-counter]* anula
            self.LR_obj_const.append(list_Constr2[i-counter])
        counter=counter+len(list_Constr2)
# DOS
        anula=1
#        if self.resource_i != 1 :
#            anula=0
#        for i in range(counter,counter+len(list_Constr3)):
#            if anula == 1:
#                self.index_L[i] = 1
#            self.lambda1[i] = self.lambda1[i] * anula
#            self.LR_obj = self.LR_obj + self.lambda1[i] * list_Constr3[i-counter]* anula
#            self.LR_obj_const.append(list_Constr3[i-counter])
#        counter=counter+len(list_Constr3)
# TRES
#        anula=1
#        if self.resource_i != 2 :
#            anula=0
#        for i in range(counter,counter+len(list_Constr4)):
#            if anula == 1:
#                self.index_L[i] = 1
#            self.lambda1[i] = self.lambda1[i] * anula
#            self.LR_obj = self.LR_obj + self.lambda1[i] * list_Constr4[i-counter]* anula
#            self.LR_obj_const.append(list_Constr4[i-counter])

        self.m.setObjective( self.function_obj + self.LR_obj, self.sense_opt)

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
