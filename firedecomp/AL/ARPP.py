"""Module with lagrangian decomposition."""

# Python packages
import gurobipy
import logging as log
import pandas as pd

# Package modules
from firedecomp.classes import solution
from firedecomp import config
from firedecomp.original import model


################################################################################
# CLASS RelaxedPrimalProblem()
################################################################################
class RelaxedPrimalProblem(model.InputModel):
    def __init__(self, problem_data, lambda1, beta1, relaxed=False, min_res_penalty=1000000,
                 valid_constraints=None,  list_y=[]):
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")
        self.v = 1
        self.lambda1 = lambda1.copy()
        self.beta = beta1.copy()
        self.problem_data = problem_data.copy_problem()
        self.relaxed = relaxed
        self.min_res_penalty = min_res_penalty
        self.NL = len(self.lambda1)
        self.valid_constraints = valid_constraints
        # Extract data from problem data
        self.__extract_data_problem__()
        if len(list_y) == 0:
            self.list_y_init = list_y.copy()
        else:
            self.list_y_init = []

        self.slack_containment = False
        # Create gurobi model
        self.model = self.__get_model__()

    ################################################################################
    # PRIVATE METHOD: __extract_data_problem__()
    ################################################################################
    def __extract_data_problem__(self):
        """ Extract data from problem class to RPP class.
        """
        # SET
        self.__extract_set_data_problem__()
        # PARAMETERS
        self.__extract_parameters_data_problem__()
        # OTHER PARAMETERS
        self.Mp = self.min_res_penalty
        self.M = sum([v for k, v in self.PER.items()])
        self.min_t = int(min(self.T))
        self.max_t = int(max(self.T))
        self.sizey = self.T + [self.min_t - 1]

        # SENSE
        self.__sense_opt__()

    ################################################################################
    # PRIVATE METHOD: __extract_set_data_problem__()
    ################################################################################
    def __extract_set_data_problem__(self):
        """ Extract SET fields from data problem
        """
        #  SETS
        self.I = self.problem_data.get_names("resources")
        self.G = self.problem_data.get_names("groups")
        self.T = self.problem_data.get_names("wildfire")
        self.Ig = {
            k: [e.name for e in v]
            for k, v in self.problem_data.groups.get_info('resources').items()}
        self.T_int = self.problem_data.wildfire

    ################################################################################
    # PRIVATE METHOD: __extract_parameters_data_problem__()
    ################################################################################
    def __extract_parameters_data_problem__(self):
        """ Extract PARAMETERS fields from data problem
        """
        self.C = self.problem_data.resources.get_info("variable_cost")
        self.P = self.problem_data.resources.get_info("fix_cost")
        self.BPR = self.problem_data.resources.get_info("performance")
        self.A = self.problem_data.resources.get_info("arrival")
        self.CWP = self.problem_data.resources.get_info("work")
        self.CRP = self.problem_data.resources.get_info("rest")
        self.CUP = self.problem_data.resources.get_info("total_work")
        self.ITW = self.problem_data.resources.get_info("working_this_wildfire")
        self.IOW = self.problem_data.resources.get_info("working_other_wildfire")
        self.TRP = self.problem_data.resources.get_info("time_between_rests")
        self.WP = self.problem_data.resources.get_info("max_work_time")
        self.RP = self.problem_data.resources.get_info("necessary_rest_time")
        self.UP = self.problem_data.resources.get_info("max_work_daily")
        self.PR = self.problem_data.resources_wildfire.get_info("resource_performance")
        self.PER = self.problem_data.wildfire.get_info("increment_perimeter")
        self.NVC = self.problem_data.wildfire.get_info("increment_cost")
        self.nMin = self.problem_data.groups_wildfire.get_info("min_res_groups")
        self.nMax = self.problem_data.groups_wildfire.get_info("max_res_groups")

    ################################################################################
    # PRIVATE METHOD: __sense_opt__()
    ################################################################################
    def __sense_opt__(self):
        self.sense_opt = gurobipy.GRB.MINIMIZE

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
        self.__create_gurobi_model__()
        self.__create_vars__()
        self.__create_objfunc__()
        self.__create_constraints_aux__
        self.__create_constraints__()
        if self.list_y_init is not None:
            self.s[self.I[0],1].lb = 1

        self.m.update()
        model = solution.Solution(
            self.m, dict(s=self.s, tr=self.tr, r=self.r, er=self.er, e=self.e, u=self.u,
                         w=self.w, z=self.z, cr=self.cr, y=self.y, mu=self.mu))
        return model

    ##########################################################################################
    # PRIVATE METHOD: __create_model__
    ################################################################################
    def __create_gurobi_model__(self):
        """Create gurobi model.
        """
        self.m = gurobipy.Model("Relaxed_Primal_Problem_LR")

    ################################################################################
    # PRIVATE METHOD: __relaxed_config__
    ################################################################################
    def __relaxed_config__(self):
        if self.relaxed is True:
            self.vtype = gurobipy.GRB.CONTINUOUS
            self.lb = 0
            self.ub = 1
        else:
            self.vtype = gurobipy.GRB.BINARY
            self.lb = 0
            self.ub = 1

    ################################################################################
    # PRIVATE METHOD: __create_vars__
    ################################################################################
    def __create_vars__(self):
        """Create vars in gurobi model class

        Args:
            gurobipy (:obj:`dict`): dictionary with attribute information to
                update.
        """
        self.__relaxed_config__()
        # VARIABLES
        # (1) Resources
        # ---------
        self.s = self.m.addVars(self.I, self.T, vtype=self.vtype, lb=self.lb, ub=self.ub, name="start")
        self.tr = self.m.addVars(self.I, self.T, vtype=self.vtype, lb=self.lb, ub=self.ub, name="travel")
        self.r = self.m.addVars(self.I, self.T, vtype=self.vtype, lb=self.lb, ub=self.ub, name="rest")
        self.er = self.m.addVars(self.I, self.T, vtype=self.vtype, lb=self.lb, ub=self.ub, name="end_rest")
        self.e = self.m.addVars(self.I, self.T, vtype=self.vtype, lb=self.lb, ub=self.ub, name="end")

        self.__create_auxiliar_vars__()
        self.__create_var_y_and_aux_vars__()

        # (3) Wildfire
        # --------
        self.mu = self.m.addVars(self.G, self.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0,
                                 name="missing_resources")

        if self.slack_containment:
            self.slack_cont1 = self.m.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, name="slack_containment_1")
            self.slack_cont2 = self.m.addVars(self.T, vtype=gurobipy.GRB.CONTINUOUS, lb=0, name="slack_containment_2")

    ################################################################################
    # PRIVATE METHOD: __create_var_y_and_aux_vars__
    ################################################################################
    def __create_var_y_and_aux_vars__(self):
        self.sizey = len(self.T + [self.min_t - 1])
        self.y = self.m.addVars(self.T + [self.min_t - 1], vtype=self.vtype, lb=self.lb, ub=self.ub,
                                name="contention")

        self.aux_total = self.m.addVars(self.NL, vtype=gurobipy.GRB.CONTINUOUS, name="aux_total_AL")
        # fixed y
        if len(self.list_y_init) > 0:
            for i in range(0, self.sizey):
                self.y[i].UB = self.list_y_init[i]
                self.y[i].LB = self.list_y_init[i]
                self.y[i].start = self.list_y_init[i]
    ################################################################################
    # PRIVATE METHOD: create_auxiliar_vars
    ################################################################################
    def __create_auxiliar_vars__(self):
        self.u = {
            (i, t):
                self.s.sum(i, self.T_int.get_names(p_max=t))
                - self.e.sum(i, self.T_int.get_names(p_max=t - 1))
            for i in self.I for t in self.T}
        self.w = {(i, t): self.u[i, t] - self.r[i, t] - self.tr[i, t] for i in self.I for t in self.T}
        self.z = {i: self.e.sum(i, '*') for i in self.I}

        self.cr = {(i, t):
            sum([
                (t + 1 - t1) * self.s[i, t1]
                - (t - t1) * self.e[i, t1]
                - self.r[i, t1]
                - self.WP[i] * self.er[i, t1]
                for t1 in self.T_int.get_names(p_max=t)])
            for i in self.I for t in self.T
            if not self.ITW[i] and not self.IOW[i]}

        self.cr.update({
            (i, t):
                (t + self.CWP[i] - self.CRP[i]) * self.s[i, self.min_t]
                + sum([
                    (t + 1 - t1 + self.WP[i]) * self.s[i, t1]
                    for t1 in self.T_int.get_names(p_min=self.min_t + 1, p_max=t)])
                - sum([
                    (t - t1) * self.e[i, t1]
                    + self.r[i, t1]
                    + self.WP[i] * self.er[i, t1]
                    for t1 in self.T_int.get_names(p_max=t)])
            for i in self.I for t in self.T
            if self.ITW[i] or self.IOW[i]})

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


        self.list_Constr = Constr1 + list_Constr2 #+ list_Constr3 + list_Constr4

        # Objective
        # =========
        self.function_obj = (sum([self.C[i] * self.u[i, t] for i in self.I for t in self.T]) +
                             sum([self.P[i] * self.z[i] for i in self.I]) +
                             sum([self.NVC[t] * self.y[t - 1] for t in self.T]) +
                             sum([self.Mp * self.mu[g, t] for g in self.G for t in self.T]) +
                             0.001 * self.y[self.max_t])

        self.function_obj_total = self.function_obj

        self.LR_obj = 0
        self.LR_obj_const = []

        for i in range(0, len(self.list_Constr)):
            Constr1 = self.list_Constr[i]
            self.LR_obj = self.LR_obj + 1 / (2 * self.beta[i]) * (
                        self.aux_total[i] * self.aux_total[i] - self.lambda1[i] * self.lambda1[i])
            self.LR_obj_const.append(Constr1)

        self.function_obj_total_pen = self.function_obj_total + self.LR_obj
        self.m.setObjective(self.function_obj, self.sense_opt)


    ################################################################################
    # PRIVATE METHOD: __create_constraints__()
    ################################################################################
    def __create_constraints__(self):
        # Constraints
        # ===========
        # Wildfire Containment
        # --------------------

        self.m.addConstr(self.y[self.min_t-1] == 1, name='start_no_contained')

        #self.m.addConstr(sum([self.PER[t]*self.y[t-1] for t in self.T]) -
        #        sum([self.PR[i, t]*self.w[i, t] for i in self.I for t in self.T]) <= 0,
        #        name='wildfire_containment_1')

        #self.m.addConstrs(
        #        (-1.0*self.M*self.y[t] + sum([self.PER[t1] for t1 in self.T_int.get_names(p_max=t)])*self.y[t-1]
        #        - sum([self.PR[i, t1]*self.w[i, t1] for i in self.I for t1 in self.T_int.get_names(p_max=t)])
        #        <= 0 for t in self.T), name='wildfire_containment_2')

        # Start of activity
        # -----------------
        self.m.addConstrs(
            (self.A[i] * self.w[i, t] <=
             sum([self.tr[i, t1] for t1 in self.T_int.get_names(p_max=t)])
             for i in self.I for t in self.T),
            name='start_activity_1')

        self.m.addConstrs(
            (self.s[i, self.min_t] +
             sum([(self.max_t + 1) * self.s[i, t]
                  for t in self.T_int.get_names(p_min=self.min_t + 1)]) <=
             self.max_t * self.z[i]
             for i in self.I if self.ITW[i] == True),
            name='start_activity_2')

        self.m.addConstrs(
            (sum([self.s[i, t] for t in self.T]) <= self.z[i]
             for i in self.I if self.ITW[i] == False),
            name='start_activity_3')

        # End of Activity
        # ---------------
        self.m.addConstrs(
            (sum([self.tr[i, t1] for t1 in self.T_int.get_names(p_min=t - self.TRP[i] + 1,
                                                                p_max=t)
                  ]) >= self.TRP[i] * self.e[i, t]
             for i in self.I for t in self.T),
            name='end_activity')

        # Breaks
        # ------
        self.m.addConstrs(
            (0 <= self.cr[i, t]
             for i in self.I for t in self.T),
            name='Breaks_1_lb')

        self.m.addConstrs(
            (self.cr[i, t] <= self.WP[i]
             for i in self.I for t in self.T),
            name='Breaks_1_ub')

        self.m.addConstrs(
            (self.r[i, t] <= sum([self.er[i, t1]
                                  for t1 in self.T_int.get_names(p_min=t,
                                                                 p_max=t + self.RP[i] - 1)])
             for i in self.I for t in self.T),
            name='Breaks_2')

        self.m.addConstrs(
            (sum([
                self.r[i, t1]
                for t1 in self.T_int.get_names(p_min=t - self.RP[i] + 1, p_max=t)]) >=
             self.RP[i] * self.er[i, t]
             if t >= self.min_t - 1 + self.RP[i] else
             self.CRP[i] * self.s[i, self.min_t] +
             sum([self.r[i, t1] for t1 in self.T_int.get_names(p_max=t)]) >=
             self.RP[i] * self.er[i, t]
             for i in self.I for t in self.T),
            name='Breaks_3')

        self.m.addConstrs(
            (sum([self.r[i, t1] + self.tr[i, t1]
                  for t1 in self.T_int.get_names(p_min=t - self.TRP[i],
                                                 p_max=t + self.TRP[i])]) >=
             len(self.T_int.get_names(p_min=t - self.TRP[i],
                                      p_max=t + self.TRP[i])) * self.r[i, t]
             for i in self.I for t in self.T),
            name='Breaks_4')

        # Maximum Number of Usage Periods in a Day
        # ----------------------------------------
        self.m.addConstrs(
            (sum([self.u[i, t] for t in self.T]) <= self.UP[i] - self.CUP[i]
             for i in self.I),
            name='max_usage_periods')
        ################################################################
        # Non-Negligence of Fronts
        # ------------------------
        self.m.addConstrs(
            (sum([self.w[i, t] for i in self.Ig[g]]) >= self.nMin[g, t]*self.y[t-1] - self.mu[g, t]
             for g in self.G for t in self.T),
            name='non-negligence_1')

        self.m.addConstrs(
            (sum([self.w[i, t] for i in self.Ig[g]]) <= self.nMax[g, t]*self.y[t-1]
             for g in self.G for t in self.T),
            name='non-negligence_2')

        ################################################################

        # Logical constraints
        # ------------------------
        self.m.addConstrs(
            (sum([t * self.e[i, t] for t in self.T]) >= sum([t * self.s[i, t] for t in self.T])
             for i in self.I),
            name='logical_1')

        self.m.addConstrs(
            (sum([self.e[i, t] for t in self.T]) <= 1
             for i in self.I),
            name='logical_2')

        self.m.addConstrs(
            (self.r[i, t] + self.tr[i, t] <= self.u[i, t]
             for i in self.I for t in self.T),
            name='logical_3')

        self.m.addConstrs(
            (sum([self.w[i, t] for t in self.T]) >= self.z[i]
             for i in self.I),
            name='logical_4')

        #self.m.update()

    ################################################################################
    # METHOD: return_LR_obj()
    ################################################################################
    def return_LR_obj(self):
        return self.LR_obj.getValue()

    ################################################################################
    # METHOD: return_LR_obj2()
    ################################################################################
    def return_LR_obj2(self):
        total = []
        num = len(self.LR_obj_const)
        for i in range(0, num):
            total.append(self.LR_obj_const[i].getValue())
        return total

    ################################################################################
    # METHOD: return_function_obj()
    ################################################################################
    def return_function_obj(self):
        return self.function_obj.getValue()

    ################################################################################
    # METHOD: return_sizey
    ################################################################################
    def return_sizey(self):
        return len(self.T + [self.min_t - 1])

    ################################################################################
    # METHOD: return_function_obj_original()
    ################################################################################
    def return_function_obj_total(self):
        return self.function_obj_total.getValue()

    ################################################################################
    # METHOD: return_function_obj_original()
    ################################################################################
    def return_function_obj_total_pen(self):
        return self.function_obj_total_pen.getValue()

    def __create_constraints_aux__(self):
        for i in range(0, len(self.list_Constr)):
            Constr1 = self.list_Constr[i]
            self.m.addConstr(self.aux_total[i] >= 0.0, name="aux1_const"+str(i))
            self.m.addConstr(self.lambda1[i] + self.beta[i] * Constr1 / self.v <= self.aux_total[i],  name="aux2_const"+str(i))
        self.m.update()