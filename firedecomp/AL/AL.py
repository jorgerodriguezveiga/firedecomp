"""Module with lagrangian decomposition methods."""
# Python packages

# Package modules
import logging as log
from firedecomp.AL import ARPP
from firedecomp.AL import ADPP
from firedecomp.fix_work import utils as _utils
from firedecomp.original import model as _model
from firedecomp.classes import problem as _problem

import time
import math
import gurobipy
import copy


###############################################################################
# CLASS LagrangianRelaxation()
###############################################################################
class AugmentedLagrangian(object):
    def __init__(
            self,
            problem_data,
            min_res_penalty=1000000,
            valid_constraints=None,
            gap=0.01,
            max_iters=100000,
            max_time=10,
            log_level="AL",
            solver_options=None,
    ):

        """Initialize the Lagrangian Relaxation object.

        Args:
            problem_data (:obj:`Problem`): problem data.
            min_res_penalty (:obj:`int`):
            gap (:obj:`float`): GAP tolerance for stop criteria.
                Defaults to 0.01.
            max_iters (:obj:`int`): maximum number of iterations. Defaults to
                10.
            max_time (:obj:`float`): maximum cpu time (in seconds). Defaults to
                3600.
            log_level (:obj:`str`): logging level. Defaults to ``'benders'``.
        """

        # PROBLEM DATA
        if problem_data.period_unit is False:
            raise ValueError("Time unit of the problem is not a period.")
        self.problem_data = problem_data
        self.solution_best = None  # index of DPP best solution
        self.solution_best_original = None
        # GLOBAL VARIABLES
        self.max_iters = max_iters
        self.max_time = max_time
        self.init_time = time.time()
        self.v = 1  # iterations
        self.NL = (1 + len(problem_data.get_names("wildfire"))) #+
                #len(problem_data.get_names("wildfire"))*len(problem_data.get_names("groups"))*2);
        # GUROBI OPTIONS
        if solver_options is None:
            solver_options = {
                'OutputFlag': 0,
                'LogToConsole': 0,
            }
        self.solver_options = solver_options

        # PARAMETERS INNER METHODS
        self.change = []
        self.beta_matrix = []
        self.lambda_matrix = []
        self.lambda_matrix_prev = []
        self.upperbound_matrix = []
        self.lobj_global = float("-inf")
        self.fobj_global = float("inf")
        self.infeas_global = float("inf")
        self.subgradient_global = []
        self.penalties_global = []
        self.index_best = -1
        self.lambda_min = 1
        self.lambda_max = 1e4
        self.lamdba_init = 1e2
        self.th_sol = 10

        # INITIALIZE RPP PROBLEM (optional)
        self.lambda1_RRP = []
        self.beta_RRP = []
        for i in range(0, self.NL):
            self.lambda1_RRP.append(self.lambda_max)
            self.beta_RRP.append(0.5)
        # Create Relaxed Primal Problem
        self.problem_RPP = ARPP.RelaxedPrimalProblem(problem_data, self.lambda1_RRP, self.beta_RRP)
        #self.solution_RPP = float("inf")
        #solver_options_RPP = {
        #    'OutputFlag': 1,
        #    'LogToConsole': 1,
        #}
        #print("SOLVE RPP")
        #self.problem_RPP.solve(solver_options_RPP)
        #print("END SOLVE RPP")
        # INITIALIZE DPP PROBLEM
        # Initialize Decomposite Primal Problem Variables
        self.problem_DPP = []
        self.N = len(self.problem_RPP.I)
        self.y_master_size = self.problem_RPP.return_sizey()
        self.counterh_matrix = []

        self.lobj_local = []
        self.lobj_local_prev = []
        self.fobj_local = []
        self.infeas_local = []
        self.subgradient_local = []
        self.subgradient_local_prev = []
        self.penalties_local = []
        self.termination_counter = []
        self.best_list_y = []

        for i in range(0, self.y_master_size):
            self.termination_counter.append(0)
            self.lobj_local.append(float("inf"))
            self.lobj_local_prev.append(float("inf"))
            self.fobj_local.append(float("inf"))
            self.infeas_local.append(float("inf"))
            self.subgradient_local.append([])
            self.penalties_local.append([])
            self.upperbound_matrix.append(float("inf"))
        self.termination_counter[self.y_master_size - 1] = self.th_sol + 1

        # INITIALIZE LAMBDA AND BETA
        for i in range(0, self.y_master_size):
            lambda_row = []
            lambda_row_prev = []
            lambda_row_inf = []
            beta_row = []
            change_row = []
            subgradient_prev_row = []
            for j in range(0, self.NL):
                lambda_row.append(self.lamdba_init)
                lambda_row_inf.append(float("inf"))
                beta_row.append(0.3)
                change_row.append(1.0)
                lambda_row_prev.append(self.lamdba_init)
                subgradient_prev_row.append(0)
            self.subgradient_local_prev.append(subgradient_prev_row)
            self.lambda_matrix.append(lambda_row)
            self.lambda_matrix_prev.append(lambda_row_prev)
            self.beta_matrix.append(beta_row)
            self.change.append(change_row)

        # CREATE ORIGINAL Problems list
        _utils.get_initial_sol(self.problem_data)
        dict_update = self.problem_data.get_variables_solution()
        print("UPDATE original problem")
        for i in range(0, self.y_master_size - 1):
            self.y_master = dict([(p, 1) for p in range(0, self.y_master_size)])
            for p in range(self.y_master_size - (1 + i), self.y_master_size):
                self.y_master[p] = 0
            print("Create index: " + str(i) + " y: " + str(self.y_master))
            model_DPP = ADPP.DecomposedPrimalProblem(self.problem_data,
                                                     self.lambda_matrix[i], self.beta_matrix[i],
                                                     self.y_master, self.N,
                                                     init_sol_dict=dict_update,
                                                     #init_sol_prob=self.problem_RPP,
                                                     min_res_penalty=min_res_penalty,
                                                     valid_constraints=valid_constraints)
            self.problem_DPP.append(model_DPP)

    ###############################################################################
    # PUBLIC METHOD subgradient()
    ###############################################################################
    def subgradient(self, subgradient, subgradient_prev, lambda_vector, beta_vector, lambda_matrix_prev, change_per, ii):

        lambda_old = lambda_vector.copy()
        beta_old = beta_vector.copy()

        stuck=0
        if max(subgradient_prev) < 0 and max(subgradient) > 0:
            stuck=1
            #print(subgradient)
            #print(subgradient_prev)

        #if ii == 5:
            #print("subgradient"+str(max(subgradient)))
            #print("subgradient_prev"+str(max(subgradient_prev)))
            #print(stuck)

        for i in range(0, self.NL):
            LRpen = subgradient[i]
            if stuck == 1 and LRpen < 0:
                new_lambda = lambda_matrix_prev[i]
            else:
                new_lambda = (lambda_old[i] + LRpen * beta_old[i])

            lambda_vector[i] = min(max(self.lambda_min, new_lambda), self.lambda_max)
            beta_vector[i] = beta_vector[i] * 1.2
            #if ii == 5:
            #print(str(LRpen)+" -> lambda "+str(lambda_old[i])+ " + "+str(beta_old[i] * LRpen) + " = " + str(lambda_vector[i]) + " update " + str(beta_old[i]) + " diff " + str(abs(abs(lambda_vector[i])-abs(lambda_old[i]))) + " beta " + str(beta_vector[i]) )#+ " change_per "+str(change_per) )
        #if ii == 5:
        #print("")
        #print("")
        for i in range(0, self.NL):
            subgradient_prev[i] = subgradient[i]
            lambda_matrix_prev[i] = lambda_old[i]
        del lambda_old
        del beta_old

    ###############################################################################
    # PUBLIC METHOD convergence_checking()
    ###############################################################################
    def convergence_checking(self):
        stop = bool(False)
        result = 0
        optimal_solution_found = 0

        # print("TERMINATION COUNTER"+str(self.termination_counter))

        # CHECK PREVIOUS LAMBDAS CHANGES
        for i in range(0, len(self.lambda_matrix) - 1):
            if self.termination_counter[i] < self.th_sol:
                lobj_diff = abs(
                    (abs(self.lobj_local[i]) - abs(self.lobj_local_prev[i])) / abs(self.lobj_local[i])) * 100
                # print(str(i) + "self.lobj_local[i] - self.lobj_local_prev[i] " + str(lobj_diff) + "% ")
                if (lobj_diff < 0.1):
                    self.termination_counter[i] = self.termination_counter[i] + 1
                else:
                    self.termination_counter[i] = 0
                self.lobj_local_prev[i] = self.lobj_local[i]

        # CHECK TERMINATION COUNTER MATRIX
        counter = 0
        all_termination_counter_finished = 0
        for i in range(0, self.y_master_size):
            if self.termination_counter[i] >= (self.th_sol):
                counter = counter + 1
        if counter == self.y_master_size:
            all_termination_counter_finished = 1
        print("counter"+str(counter)+" termination_counter"+str(self.y_master_size))

        # STOPPING CRITERIA CASES
        current_time = time.time() - self.init_time
        # check convergence
        if (self.v >= self.max_iters):
            print("[STOP] Max iters achieved!")
            stop = bool(True)
        if (current_time >= self.max_time):
            print("[STOP] Max execution time achieved!")
            stop = bool(True)
        elif (all_termination_counter_finished == 1):
            print("[STOP] Convergence achieved, optimal local point searched!")
            stop = bool(True)
        elif (optimal_solution_found == 1):
            print("[STOP] Convergence achieved, optimal solution found!")
            stop = bool(True)

        return stop

    ###############################################################################
    # PUBLIC METHOD solve()
    ###############################################################################
    # OLLO FACELO POR GRUPO DE PERIODOS, POR EXEMPLO DE CINCO EN CINCO
    ###############################################################################
    def solve(self):
        print("SOLVE ALGORITHM")
        termination_criteria = bool(False)

        while termination_criteria == False:
            # (1) Solve DPP problems
            for i in range(0, self.y_master_size - 1):
                # Show iteration results
                if i == 0:
                    log.info("Iteration # mi lambda f(x) L(x,mi,lambda) penL")
                    print("\n\nIter: " + str(self.v) + " " +
                          "LR(x): " + str(self.lobj_global) + " " +
                          "f(x):" + str(self.fobj_global) + " " +
                          "penL:" + str(self.infeas_global) + "\n")
                if self.termination_counter[i] < self.th_sol:
                    print("### Y -> " + str(self.problem_DPP[i].list_y))
                    DPP_sol_row = []
                    DPP_sol_unfeasible = False
                    total_obj_function = []
                    total_unfeasibility = []
                    total_subgradient = []
                    total_obj_function_pen = []
                    total_problem = []
                    self.lobj_local[i] = 0
                    self.fobj_local[i] = 0
                    self.subgradient_local[i] = []
                    for z in range(0, self.NL):
                        self.subgradient_local[i].append(float("-inf"))
                    self.penalties_local[i] = []
                    self.infeas_local[i] = 0
                    #for j in range(0, self.N):
                    j=0
                    try:
                        self.problem_DPP[i].m.reset(0)
                        self.problem_DPP[i].change_resource(j, self.lambda_matrix[i], self.beta_matrix[i], self.v)
                        DPP_sol_row.append(self.problem_DPP[i].solve(self.solver_options))

                        if (DPP_sol_row[j].model.Status == 3) or (DPP_sol_row[j].model.Status == 4):
                                DPP_sol_unfeasible = True
                                break
                    except:
                        print("Error Solver: Lambda/beta error")
                        DPP_sol_unfeasible = True
                        break
                    total_problem.append(self.problem_DPP[i].problem_data.copy_problem())
                    subgradient = self.problem_DPP[i].return_LR_obj2()
                    total_obj_function_pen.append(self.problem_DPP[i].return_function_obj_total_pen())
                    total_obj_function.append(self.problem_DPP[i].return_function_obj_total())
                    total_unfeasibility.append(max(subgradient))
                    total_subgradient.append(subgradient)

                    if DPP_sol_unfeasible:
                        self.termination_counter[i] = self.th_sol + 1
                    else:
                        bestid = self.problem_DPP[i].return_best_candidate(total_obj_function,total_unfeasibility)
                        self.lobj_local[i] = total_obj_function_pen[bestid]
                        self.fobj_local[i] = total_obj_function[bestid]
                        self.infeas_local[i] = total_unfeasibility[bestid]
                        self.subgradient_local[i] = total_subgradient[bestid]
                        #if i == 5:
                        print("TOTAL" + str(i) +
                                " LR " + str(self.lobj_local[i]) +
                                " fobj " + str(self.fobj_local[i]) +
                                " Infeas " + str(self.infeas_local[i]))

                        # check UB
                        #if self.infeas_local[i] <= 0 and self.upperbound_matrix[i] > self.fobj_local[i]:
                        #    self.upperbound_matrix[i] = self.fobj_local[i]
                        # print("\tlobj "+str(self.lobj_local[i])+" fobj "+str(self.fobj_local[i])+" infeas "+str(self.infeas_local[i]))
                        # calc new lambda and beta
                        self.subgradient(self.subgradient_local[i], self.subgradient_local_prev[i],
                                         self.lambda_matrix[i], self.beta_matrix[i],
                                         self.lambda_matrix_prev[i], self.change[i], i)
                        # print("Gather_solution"+str(i))
                        #self.problem_DPP[i].update_original_values(DPP_sol_row[bestid], self.change[i], total_obj_function, total_unfeasibility)

                        if self.fobj_global > self.fobj_local[i] and (self.infeas_local[i] <= 0):  # or (self.fobj_global > self.fobj_local[i] and (self.infeas_local[i] < self.infeas_global)):
                            #print("")
                            #print("ENTRA"+str(self.infeas_local[i]))
                            #print("")
                            #print(total_problem[bestid].get_solution_info())
                            self.problem_DPP[i].problem_data = total_problem[bestid]
                            self.lobj_global = self.lobj_local[i]
                            self.fobj_global = self.fobj_local[i]
                            self.subgradient_global = self.subgradient_local[i]
                            self.infeas_global = self.infeas_local[i]

                            self.solution_best_original = total_problem[bestid].copy_problem() #self.update_problem_data_sol(self.problem_DPP[i])
                            self.solution_best_original.constrvio = self.infeas_global
                            self.solution_best_original.solve_status = 2
                            print(self.solution_best_original.get_solution_info())

                            for z in range(1, self.NL):
                                self.change[i][z] = 1
                    DPP_sol_row.clear()
            print("TERMINATION COUNTER"+str(self.termination_counter))
            # (3) Check termination criteria
            termination_criteria = self.convergence_checking()
            self.v = self.v + 1

        # DESTROY DPP
        print(self.solution_best_original.get_solution_info())
        self.problem_data = self.solution_best_original
        return self.solution_best_original

    ###############################################################################
    # PRIVATE extract_infeasibility()
    ###############################################################################
    def extract_infeasibility(self, subgradient):
        infeas = 0
        for i in range(0, len(subgradient)):
            if (subgradient[i] > 0):
                infeas = infeas + subgradient[i]
        return infeas

    ###############################################################################
    # PRIVATE destroy_DPP_set()
    ###############################################################################
    def destroy_DPP_set(self):
        for i in range(0, len(self.problem_DPP)):
            len_p = len(self.problem_DPP[i])
            if (len_p > 0):
                for j in range(0, len_p):
                    del self.problem_DPP[i][0]
                self.problem_DPP[i] = []
        self.problem_DPP = []
        # print("DESTROY")

    ###############################################################################
    # PRIVATE METHOD __log__()
    ###############################################################################
    #    def __log__(self, level="AL"):
    #        log.addLevelName(80, "AL")
    #        log.Logger.LR = logging.LR

    #        if level != 'AL':
    #            log_level = getattr(log, level)
    #            logger = log.getLogger('AL_logging')
    #            logger.setLevel(log_level)
    #            logger.addFilter(logging.LRFilter())
    #            if len(logger.handlers) == 0:
    #                ch = log.StreamHandler()
    #                ch.setLevel(log_level)
    #                # create formatter and add it to the handlers
    #                formatter = log.Formatter("%(levelname)8s: %(message)s")
    #                ch.setFormatter(formatter)
    #                logger.addHandler(ch)
    #        else:
    #            log_level = 80
    #            logger = log.getLogger('AL')
    #            logger.setLevel(log_level)
    #            if len(logger.handlers) == 0:
    #                ch = log.StreamHandler()
    #                ch.setLevel(log_level)
    #                # create formatter and add it to the handlers
    #                formatter = log.Formatter("%(message)s")
    #                ch.setFormatter(formatter)
    #                logger.addHandler(ch)
    #
    #        self.log = logger
    #        return 1

    def update_problem_data_sol(self, solution):

        problem = self.problem_data.copy_problem()
        problem = self.solution_to_problem(problem,solution)

        return problem

    def solution_to_problem(self, problem, solution):
        problem.mipgap = None
        problem.mipgapabs = None
        problem.constrvio = self.infeas_global
        problem.solve_status = 2

        variables = solution
        data = problem.data

        problem.resources.update(
            {i: {'select': round(variables.z[i].getValue()) == 1}
             for i in data.I})

        s = {(i, t): round(variables.s[i, t].x) == 1
             for i in data.I for t in data.T}
        u = {(i, t): round(variables.u[i, t].getValue()) == 1
             for i in data.I for t in data.T}
        e = {(i, t): round(variables.e[i, t].x) == 1
             for i in data.I for t in data.T}
        w = {(i, t): round(variables.w[i, t].getValue()) == 1
             for i in data.I for t in data.T}
        r = {(i, t): round(variables.r[i, t].x) == 1
             for i in data.I for t in data.T}
        er = {(i, t): round(variables.er[i, t].x) == 1
              for i in data.I for t in data.T}
        tr = {(i, t): round(variables.tr[i, t].x) == 1
              for i in data.I for t in data.T}

        problem.resources_wildfire.update(
            {(i, t): {
                'start': s[i, t],
                'use': u[i, t],
                'end': e[i, t],
                'work': w[i, t],
                'travel': tr[i, t],
                'rest': r[i, t],
                'end_rest': er[i, t]
            }
                for i in data.I for t in data.T})

        problem.groups_wildfire.update(
            {(g, t): {'num_left_resources': variables.mu[g, t].x}
             for g in data.G for t in data.T})

        contained = {t: variables.y[t].x == 0
                     for t in data.T}
        contained_period = [t for t, v in contained.items()
                            if v is True]

        if len(contained_period) > 0:
            first_contained = min(contained_period) + 1
        else:
            first_contained = data.max_t + 1

        problem.wildfire.update(
            {t: {'contained': False if t < first_contained else True}
             for t in data.T})

        return problem