"""Module with lagrangian decomposition methods."""
# Python packages
import gurobipy
import logging as log
import pandas as pd
import copy

# Package modules
from firedecomp.classes import solution
from firedecomp import config
from firedecomp.AL import ARPP


# Subproblem ------------------------------------------------------------------
class DecomposedPrimalProblem(ARPP.RelaxedPrimalProblem):
    def __init__(self, problem_data, lambda1, beta1, resource_i, list_y, sol, nproblems, NL, upperbound, relaxed=False,
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
        self.resource_i = resource_i
        self.list_y = list_y
        self.solution = sol
        self.nproblems = nproblems
        self.NL = NL
        self.UB_value = upperbound

        super().__init__(problem_data, lambda1, beta1, NL, relaxed, min_res_penalty)
        self.__update_vars__()

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
# PRIVATE METHOD: __create_var_y_and_aux_vars__
# OVERWRITE RelaxedPrimalProblem.__create_var_y_and_aux_vars__()
################################################################################
    def __create_var_y_and_aux_vars__(self):
        self.sizey = len(self.T + [self.min_t-1])
        self.y = self.m.addVars(self.T + [self.min_t-1], vtype=self.vtype,
                                    lb=self.lb, ub=self.ub, name="contention")
        self.aux_total  = self.m.addVars(self.NL,vtype=gurobipy.GRB.CONTINUOUS, name="aux_total_AL")
        self.aux_mult = self.m.addVars(self.NL,vtype=gurobipy.GRB.CONTINUOUS, name="aux_mult_AL")

        # fixed y
        for i in range(0,self.sizey):
            self.y[i].UB=self.list_y[i]
            self.y[i].LB=self.list_y[i]
            self.y[i].Start=self.list_y[i]

################################################################################
# PRIVATE METHOD: __update_vars__
# OVERWRITE RelaxedPrimalProblem.__update_vars__()
################################################################################
    def __update_vars__(self):

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
        # UPDATE MODEL
        self.__create_auxiliar_vars__()
        self.m.update()

################################################################################
# PRIVATE METHOD: __create_objfunc__()
################################################################################
    def __create_objfunc__(self):
# Wildfire Containment (2) and (3)
        Constr1 = []
        Constr1.append(sum([self.PER[t]*self.y[t-1] for t in self.T]) -
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

        list_Constr = Constr1 + list_Constr2 + list_Constr3 + list_Constr4

# Objective
# =========
        self.function_obj_total = (sum([self.C[i]*self.u[i, t] for i in self.I for t in self.T]) +
                       sum([self.P[i] * self.z[i] for i in self.I]) +
                       sum([self.NVC[t] * self.y[t-1] for t in self.T]) +
                       sum([self.Mp*self.mu[g, t] for g in self.G for t in self.T]) +
                       0.001*self.y[self.max_t])

        if (self.resource_i == 0):
            self.function_obj = (sum([self.C[self.I[self.resource_i]]*self.u[self.I[self.resource_i], t] for t in self.T]) +
                   sum([self.P[self.I[self.resource_i]] * self.z[self.I[self.resource_i]]]) +
                   sum([self.NVC[t] * self.y[t-1] for t in self.T]) +
                   sum([self.Mp*self.mu[g, t] for g in self.G for t in self.T]) +
                   0.001*self.y[self.max_t])
        else:
            self.function_obj = (sum([self.C[self.I[self.resource_i]]*self.u[self.I[self.resource_i], t] for t in self.T]) +
                       sum([self.P[self.I[self.resource_i]] * self.z[self.I[self.resource_i]]]))

        self.LR_obj = 0
        self.LR_obj_const = []

        for i in range(0, len(list_Constr)):
            Constr1 = list_Constr[i]
            self.aux_mult[i] = self.lambda1[i] + self.beta[i] * Constr1
            self.LR_obj = self.LR_obj + 1.0/(2.0*self.beta[i]) * (self.aux_total[i]*self.aux_total[i] - self.lambda1[i]*self.lambda1[i])
            self.LR_obj_const.append(Constr1)

        self.m.setObjective( self.function_obj + self.LR_obj, self.sense_opt)


################################################################################
# METHOD: UPDATE MODEL
################################################################################
    def update_model(self, lambda1, beta1, sol, ub):
        """ Update lambda in DPP model
            Args:
            lambda1 (:obj:`list`): Array with lambda values.
        """

        self.solution = sol
        self.lambda1 = lambda1
        self.beta = beta1
        self.UB_value = ub
        self.__update_vars__()
        self.__create_objfunc__()
        self.upperbound_const.setAttr("rhs", self.UB_value)
        self.m.update()
        #rint("SIZE VARS "+str(len(self.m.getVars())))
        #print("SIZE CONS "+str(len(self.m.getConstrs())))
        #print("test "+ str(self.m.getConstrByName("upperbound_const").RHS) + " " + str(self.UB_value))
        self.model = solution.Solution(
            self.m, dict(s=self.s, tr=self.tr, r=self.r, er=self.er, e=self.e, u=self.u,
            w=self.w, z=self.z, cr=self.cr, y=self.y, mu=self.mu))
