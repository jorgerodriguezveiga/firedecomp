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
    def __init__(self, problem_data, lambda1, beta1, list_y, nproblems, init_sol_dict=None, init_sol_prob=None, relaxed=False,
                 min_res_penalty=1000000, valid_constraints=None):
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
        self.list_y = list_y.copy()
        self.nproblems = nproblems
        self.NL = len(lambda1)
        self.resource_i = 0
        super().__init__(problem_data, lambda1, beta1, relaxed, min_res_penalty, valid_constraints)
        self.__create_constraints_aux__()
        self.copy_original_bounds()
        solver_options = {
            'OutputFlag': 0,
            'LogToConsole': 0,
            'TimeLimit': 5,
        }
        self.solve(solver_options=solver_options)
        self.set_init_value_prob(self)
        #if init_sol_dict is not None:
        #    self.set_init_value_dict(init_sol_dict)
        #if init_sol_prob is not None:
        #    self.set_init_value_prob(init_sol_prob)
    ##########################################################################################
    # PRIVATE METHOD: __create_gurobi_model__ ()
    # OVERWRITE RelaxedPrimalProblem.__create_gurobi_model__()
    def __create_gurobi_model__(self):
        """Create gurobi model.
        """
        self.m = gurobipy.Model("Decomposed_Primal_Problem")

    ################################################################################
    # PRIVATE METHOD: __create_var_y_and_aux_vars__
    # OVERWRITE RelaxedPrimalProblem.__create_var_y_and_aux_vars__()
    ################################################################################
    def __create_var_y_and_aux_vars__(self):
        self.sizey = len(self.T + [self.min_t - 1])
        self.y = self.m.addVars(self.T + [self.min_t - 1], vtype=self.vtype,
                                lb=self.lb, ub=self.ub, name="contention")
        self.aux_total = self.m.addVars(self.NL, vtype=gurobipy.GRB.CONTINUOUS, name="aux_total_AL")

        # fixed y
        for i in range(0, self.sizey):
            self.y[i].UB = self.list_y[i]
            self.y[i].LB = self.list_y[i]
            self.y[i].start = self.list_y[i]

    ################################################################################
    # PRIVATE METHOD: __create_objfunc__()
    ################################################################################
    def __create_objfunc__(self):
        # Wildfire Containment (2) and (3)
        Constr1 = []
        Constr1.append(sum([self.PER[t] * self.y[t - 1] for t in self.T]) -
                       sum([self.PR[i, t] * self.w[i, t]
                            for i in self.I for t in self.T]))

        list_Constr2 = list(
            -1.0 * self.M * self.y[t] + sum([self.PER[t1] for t1 in self.T_int.get_names(p_max=t)]) * self.y[t - 1]
            - sum([self.PR[i, t1] * self.w[i, t1] for i in self.I for t1 in self.T_int.get_names(p_max=t)])
            for t in self.T)

        # Non-Negligence of Fronts (14) and (15)
        #list_Constr3 = list(
        #    (-1.0 * sum([self.w[i, t] for i in self.Ig[g]])) - (self.nMin[g, t] * self.y[t - 1] + self.mu[g, t])
        #    for g in self.G for t in self.T)
        #list_Constr4 = list(sum([self.w[i, t] for i in self.Ig[g]]) - self.nMax[g, t] * self.y[t - 1]
        #                    for g in self.G for t in self.T)

        self.list_Constr = Constr1 + list_Constr2  #+ list_Constr3 + list_Constr4

        # Objective
        # =========
        self.function_obj_total = (sum([self.C[i] * self.u[i, t] for i in self.I for t in self.T]) +
                                   sum([self.P[i] * self.z[i] for i in self.I]) +
                                   sum([self.NVC[t] * self.y[t - 1] for t in self.T]) +
                                   sum([self.Mp * self.mu[g, t] for g in self.G for t in self.T]) +
                                   0.001 * self.y[self.max_t])
        if self.resource_i == 0:
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

        for i in range(0, len(self.list_Constr)):
            Constr1 = self.list_Constr[i]
            self.LR_obj = self.LR_obj + 1.0 / (2.0 * self.beta[i]) * (
                    self.aux_total[i] * self.aux_total[i] - self.lambda1[i] * self.lambda1[i])
            self.LR_obj_const.append(Constr1)

        self.function_obj_total_pen = self.function_obj_total + self.LR_obj
        self.m.setObjective(self.function_obj + self.LR_obj, self.sense_opt)



    ################################################################################
    # METHOD: set_init_value
    ################################################################################
    def set_init_value_dict(self, dict_update):
        """ Update lambda in DPP model
            Args:
            lambda1 (:obj:`list`): Array with lambda values.
        """
        Tlen = self.T
        Glen = self.G
        for i in range(0, len(self.I)):
            res = self.I[i]
            for tt in Tlen:
                self.s_original[res, tt] = round(dict_update.get('s')[res, tt])
                self.tr_original[res, tt] = round(dict_update.get('tr')[res, tt])
                self.r_original[res, tt] = round(dict_update.get('r')[res, tt])
                self.er_original[res, tt] = round(dict_update.get('er')[res, tt])
                self.e_original[res, tt] = round(dict_update.get('e')[res, tt])

        for grp in Glen:
            for tt in Tlen:
                self.mu_original[grp, tt] = dict_update.get('mu')[grp, tt]

    ################################################################################
    # METHOD: set_init_value
    ################################################################################
    def set_init_value_prob(self, problem_update):
        """ Update lambda in DPP model
            Args:
            lambda1 (:obj:`list`): Array with lambda values.
        """
        Tlen = self.T
        Glen = self.G
        for i in range(0, len(self.I)):
            res = self.I[i]
            for tt in Tlen:
                self.s_original[res, tt] = round(problem_update.s[res, tt].X)
                self.tr_original[res, tt] =round(problem_update.tr[res, tt].X)
                self.r_original[res, tt] = round(problem_update.r[res, tt].X)
                self.er_original[res, tt] =round(problem_update.er[res, tt].X)
                self.e_original[res, tt] = round(problem_update.e[res, tt].X)

        for grp in Glen:
            for tt in Tlen:
                self.mu_original[grp, tt] = problem_update.mu[grp, tt].X

    ################################################################################
    # METHOD: copy_original_bounds
    ################################################################################
    def copy_original_bounds(self):
        """ Copy original dict
             Args:
             lambda1 (:obj:`list`): Array with lambda values.
         """
        self.s_original = {}
        self.tr_original = {}
        self.r_original = {}
        self.er_original = {}
        self.e_original = {}
        self.mu_original = {}

    ################################################################################
    # METHOD: change_resource
    ################################################################################
    def change_resource(self, resource_i, lambda1, beta1, v):
        self.lambda1 = lambda1.copy()
        self.beta = beta1.copy()
        self.resource_i = resource_i
        self.v = v
        Tlen = self.T
        Glen = self.G

        self.m.reset()

        for i in range(0, len(self.I)):
            res = self.I[i]
            for tt in Tlen:
                if self.resource_i != i:
                    self.s[res, tt].setAttr("ub", round(self.s_original[res, tt]))
                    self.tr[res, tt].setAttr("ub", round(self.tr_original[res, tt]))
                    #self.r[res, tt].setAttr("ub", round(self.r_original[res, tt]))
                    self.er[res, tt].setAttr("ub", round(self.er_original[res, tt]))
                    self.e[res, tt].setAttr("ub", round(self.e_original[res, tt]))
                    self.s[res, tt].setAttr("lb", round(self.s_original[res, tt]))
                    self.tr[res, tt].setAttr("lb", round(self.tr_original[res, tt]))
                    #self.r[res, tt].setAttr("lb", round(self.r_original[res, tt]))
                    self.er[res, tt].setAttr("lb", round(self.er_original[res, tt]))
                    self.e[res, tt].setAttr("lb", round(self.e_original[res, tt]))
                else:
                    self.s[res, tt].setAttr("ub", self.ub)
                    self.tr[res, tt].setAttr("ub", self.ub)
                    self.r[res, tt].setAttr("ub", self.ub)
                    self.er[res, tt].setAttr("ub", self.ub)
                    self.e[res, tt].setAttr("ub", self.ub)
                    self.s[res, tt].setAttr("lb", self.lb)
                    self.tr[res, tt].setAttr("lb", self.lb)
                    self.r[res, tt].setAttr("lb", self.lb)
                    self.er[res, tt].setAttr("lb", self.lb)
                    self.e[res, tt].setAttr("lb", self.lb)

        self.__create_objfunc__()
        self.__modify_constraints_aux__()
        self.m.update()

    def return_best_candidate(self, total_obj_function, total_unfeasibility):
        candidate_list = []
        counter = 0
        for i in range(0, len(total_unfeasibility)):
            if total_unfeasibility[i] <= 0:
                candidate_list.append(i)
        if len(candidate_list) > 0:
            counter = candidate_list[0]
            for i in range(0, len(candidate_list)):
                if total_obj_function[counter] > total_obj_function[candidate_list[i]]:
                    counter = candidate_list[i]
        else:
            counter = total_unfeasibility.index(min(total_unfeasibility))

        return counter

    ################################################################################
    # METHOD: update_original_values
    ################################################################################
    def update_original_values(self, DPP, change):

        change = 1
        if change == 1:
            Tlen = self.T
            Glen = self.G
            for res in self.problem_data.get_names("resources"):
                for tt in Tlen:
                    self.s_original[res, tt] = DPP.get_variables().get_variable('s')[res, tt].X
                    self.tr_original[res, tt] = DPP.get_variables().get_variable('tr')[res, tt].X
                    self.r_original[res, tt] = DPP.get_variables().get_variable('r')[res, tt].X
                    self.er_original[res, tt] = DPP.get_variables().get_variable('er')[res, tt].X
                    self.r_original[res, tt] = DPP.get_variables().get_variable('e')[res, tt].X

            for grp in Glen:
                for tt in Tlen:
                    self.mu_original[grp, tt] = DPP.get_variables().get_variable('mu')[grp, tt].X


    ################################################################################
    # PRIVATE METHOD: __modify_constraints_aux__()
    ################################################################################
    def __modify_constraints_aux__(self):
        for i in range(0, len(self.list_Constr)):
            Constr1 = self.list_Constr[i]
            self.m.remove(self.m.getConstrByName("aux2_const"+str(i)))
            self.m.addConstr(self.lambda1[i] + self.beta[i] * Constr1 / self.v <= self.aux_total[i],
                             name="aux2_const" + str(i))

        self.m.update()
