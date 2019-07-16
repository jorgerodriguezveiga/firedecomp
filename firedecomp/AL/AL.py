"""Module with lagrangian decomposition methods."""
# Python packages

# Package modules
import logging as log
from firedecomp import AL
from firedecomp import original
from firedecomp import logging
from firedecomp.AL import ARPP
from firedecomp.AL import ADPP
from firedecomp.AL import ARDP
from firedecomp.classes import solution as _sol
from firedecomp.fix_work import utils as _utils
from firedecomp.original import model as _model

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
        gap=0.01,
        max_iters=100000,
        max_time=60,
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

        # GLOBAL VARIABLES
        self.max_iters = max_iters
        self.max_time = max_time
        self.v = 1 # iterations
        self.NL =  1 + len(problem_data.get_names("wildfire"))

        # GUROBI OPTIONS
        if solver_options == None:
            solver_options = {
                #'MIPGap': 0.01,
                #'MIPGapAbs': 0.01,
                'OutputFlag': 0,
                'LogToConsole': 0,
            }
        self.solver_options = solver_options

        # PARAMETERS INNER METHODS
        self.beta_matrix = []
        self.lambda_matrix = []
        self.lambda_matrix_prev = []
        self.upperbound_matrix =[]
        self.lobj_global = float("-inf")
        self.fobj_global = float("inf")
        self.infeas_global = float("inf")
        self.subgradient_global = []
        self.penalties_global = []
        self.index_best = -1

        # CREATE ORIGINAL Problem
        self.original_problem = _model.InputModel(problem_data)

        # INITIALIZE RPP PROBLEM
        self.lambda1_RRP = []
        self.beta_RRP = []
        init_value=1000
        for i in range(0,self.NL):
            self.lambda1_RRP.append(init_value)
            self.beta_RRP.append(0.1)

        # Create Relaxed Primal Problem
        self.problem_RPP = ARPP.RelaxedPrimalProblem(problem_data, self.lambda1_RRP, self.beta_RRP);
        self.solution_RPP = float("inf")

        # INITIALIZE DPP PROBLEM
        # Initialize Decomposite Primal Problem Variables
        self.problem_DPP = []
        self.N = len(self.problem_RPP.I)
        self.groupR = []
        self.y_master_size = self.problem_RPP.return_sizey()
        self.counterh_matrix = []
        init_value=100

        self.lobj_local=[]
        self.lobj_local_prev=[]
        self.fobj_local=[]
        self.infeas_local=[]
        self.subgradient_local=[]
        self.penalties_local=[]
        self.upperbound_matrix=[]
        self.termination_counter = []

        for i in range(0,self.y_master_size):
            self.termination_counter.append(0)
            self.lobj_local.append(float("inf"))
            self.lobj_local_prev.append(float("inf"))
            self.fobj_local.append(float("inf"))
            self.infeas_local.append(float("inf"))
            self.subgradient_local.append([])
            self.penalties_local.append([])
            self.upperbound_matrix.append(float("inf"))

        # INITIALIZE LAMBDA AND BETA
        for i in range(0,self.y_master_size):
            lambda_row = []
            lambda_row_inf = []
            beta_row = []
            for j in range(0,self.NL):
                lambda_row.append(init_value+j)
                lambda_row_inf.append(float("inf"))
                beta_row.append(0.5)
            self.lambda_matrix.append(lambda_row)
            self.lambda_matrix_prev.append(lambda_row_inf)
            self.beta_matrix.append(beta_row)

###############################################################################
# PUBLIC METHOD subgradient()
###############################################################################
    def subgradient(self, subgradient, lambda_vector, beta_vector, ii):

        lambda_old = lambda_vector.copy()
        beta_old = beta_vector.copy()

        for i in range(0,self.NL):

            LRpen = subgradient[i]

            lambda_vector[i] = max ( 0, (lambda_old[i] + LRpen * beta_old[i])/self.v  )
            if abs(lambda_vector[i]) > 0:
                change_per = abs(abs(lambda_vector[i])-abs(lambda_old[i]))/abs(lambda_vector[i])
            else:
                change_per = 0
            if change_per >= 0.2 : #and beta_vector[i] <= 5:
                beta_vector[i] = beta_vector[i] * 1.1

            if ii == 1:
                print(str(LRpen)+" -> lambda "+str(lambda_old[i])+ " + "+str(beta_old[i] * LRpen) + " = " + str(lambda_vector[i]) + " update " + str(beta_old[i]) + " diff " + str(abs(abs(lambda_vector[i])-abs(lambda_old[i]))) + " beta " + str(beta_vector[i]) + "change_per "+str(change_per) )
        if ii == 1:
            print("")

        del lambda_old
        del beta_old

###############################################################################
# PUBLIC METHOD convergence_checking()
###############################################################################
    def convergence_checking(self):
        stop = bool(False)
        result = 0
        optimal_solution_found = 0

# CHECK PREVIOUS LAMBDAS CHANGES
        for i in range(0,len(self.lambda_matrix)-1):
            if self.termination_counter[i] < 2 :
                check = 0
                lobj_diff = abs((abs(self.lobj_local[i]) - abs(self.lobj_local_prev[i]))/abs(self.lobj_local[i]))*100
                print(str(i) + "self.lobj_local[i] - self.lobj_local_prev[i] " + str(lobj_diff) + "% ")
                if (lobj_diff < 0.1):
                    self.termination_counter[i]  = self.termination_counter[i] + 1
                else:
                    self.termination_counter[i]  = 0
                self.lobj_local_prev[i] = self.lobj_local[i]
            elif self.termination_counter[i] == 2 :
                # TEST WITH GUROBI IF IT IS A OPTIMAL SOLUTION
                solver_options = {
                    'OutputFlag': 0,
                    'LogToConsole': 0,
                }
                self.original_problem.update_model(self.DPP_sol[i])
                solution = self.original_problem.solve(solver_options=solver_options)
                print(solution.get_model().Status)
                print(solution.get_objfunction())
                print(self.fobj_local[i])
                print(i)
                if (self.original_problem.model.model.Status == 2) and (solution.get_objfunction()==self.fobj_local[i]):
                    optimal_solution_found = 1
                    self.termination_counter[i] = 3
                self.termination_counter[i]  = self.termination_counter[i] + 1

# CHECK TERMINATION COUNTER MATRIX
        counter = 0
        all_termination_counter_finished = 0
        for i in range(0,self.y_master_size):
            if self.termination_counter[i] == 3:
                counter = counter + 1
        if counter == self.y_master_size:
            all_termination_counter_finished = 1

# STOPPING CRITERIA CASES
        # check convergence
        if (self.v >= self.max_iters):
            print("[STOP] Max iters achieved!")
            stop = bool(True)
        elif (all_termination_counter_finished ==  1):
            print("[STOP] Convergence achieved, optimal local point searched!")
            stop = bool(True)
        elif (optimal_solution_found == 1):
            print("[STOP] Convergence achieved, optimal solution found!")
            stop = bool(True)
        return stop

###############################################################################
# PUBLIC METHOD solve()
###############################################################################
    def solve(self, max_iters=100):
        termination_criteria = bool(False)
        changes = 0

        # (0) Calc initial solution and create DPP problems
        self.DPP_sol = []
        init_options = {
                'IterationLimit': 1,
                'OutputFlag': 0,
                'LogToConsole': 0,
        }
        _utils.get_initial_sol(self.problem_data)
        isol = self.problem_data.get_variables_solution()
        osol = self.problem_RPP.solve(solver_options=init_options)
        initial_solution = self.create_initial_solution(isol, osol)
        DPP_sol_feasible = []
        for i in range(0,self.y_master_size-1):
            DPP_sol_feasible.append(1)
            self.DPP_sol.append(initial_solution)

        # (1) Initialize DPP
        print("CREATE SET DPP")
        self.create_DPP_set()
        print("END CREATE DPP")
        while (termination_criteria == False):
            # (1) Solve DPP problems
            for i in range(0,self.y_master_size-1):
            # Show iteration results
                if i == 1:
                    log.info("Iteration # mi lambda f(x) L(x,mi,lambda) penL")
                    print("\n\n\n\n\n\nIter: " + str(self.v) + " " +
                          "LR(x): "+ str(self.lobj_global) + " " +
                          "f(x):"  + str(self.fobj_global) + " " +
                          "penL:"  + str(self.subgradient_global) +"\n")
                print(str(i) + "- termination_counter "+str(self.termination_counter[i]))
                if self.termination_counter[i] < 2 :
                    print("### START Y -> "+str(self.problem_DPP[i][0].list_y))
                    DPP_sol_row = []
                    DPP_sol_row_feasible = []
                    stop_inf = False
                    inf_sol=0
                    self.lobj_local[i]=0
                    self.fobj_local[i]=0
                    self.subgradient_local[i] = []
                    self.penalties_local[i] = []
                    self.infeas_local[i] = 0
                    for j in range(0,self.N):
                        try:
                            DPP_sol_row.append(self.problem_DPP[i][j].solve(self.solver_options))
                            if (DPP_sol_row[j].model.Status == 3):
                                DPP_sol_row_feasible.append(0)
                                print("Error Solver: Status 3")
                            else:
                                DPP_sol_row_feasible.append(1)
                        except:
                            print("Error Solver: Lambda/beta error")
                            DPP_sol_row.append(initial_solution)
                            DPP_sol_row_feasible.append(0)

                        if (DPP_sol_row_feasible[j] == 0):
                            stop_inf = True
                            break
                        self.lobj_local[i] = self.lobj_local[i] + DPP_sol_row[j].get_objfunction()
                        self.fobj_local[i] = self.fobj_local[i] + self.problem_DPP[i][j].return_function_obj(DPP_sol_row[j])
                        self.subgradient_local[i] = self.problem_DPP[i][j].return_LR_obj2(DPP_sol_row[j])
                        self.penalties_local[i] = self.problem_DPP[i][j].return_LR_obj(DPP_sol_row[j])
                        self.infeas_local[i] = self.infeas_local[i] + self.extract_infeasibility(self.subgradient_local[i])
                        print("XResource"+  str(j)  +   " " +   str(i)+
                            " UB "       +  str(self.lobj_local[i])+
                            " fobj "     +  str(self.fobj_local[i])+
                            " Infeas "+str(self.infeas_local[i]) + "  ||  " + str(self.subgradient_local[i]))
                    # update lambda
                    for j in range(0,self.N):
                        if (DPP_sol_row_feasible[j] == 0):
                            stop_inf = True
                            self.termination_counter[i] = 3
                            break

                    # check Upper Bound
                    if self.infeas_local[i] <= 0 and self.upperbound_matrix[i] < self.fobj_local[i]:
                        self.upperbound_matrix[i] = self.fobj_local[i]


                    if (self.lobj_global < self.lobj_local[i] and (self.infeas_local[i] <= 0) and (DPP_sol_row_feasible[j] != 0)):
                        self.lobj_global  = self.lobj_local[i]
                        self.fobj_global = self.fobj_local[i]
                        self.subgradient_global = self.subgradient_local[i]
                        self.penalties_global = self.penalties_local[i]
                        self.infeas_global = self.infeas_local[i]
                        self.index_best_i = i
                        self.index_best_j = j
                        change=1

                    if (stop_inf == False):
                        print("")
                        print("")
                        print("TOTAL = UB "+str(self.lobj_local[i])+" fobj "+str(self.fobj_local[i]))
                        print("")
                        print("")

                        self.subgradient( self.subgradient_local[i] , self.lambda_matrix[i], self.beta_matrix[i], i)

                        if (inf_sol <= 0):
                            self.DPP_sol[i]=self.gather_solution(DPP_sol_row, self.DPP_sol[i])

                    else:
                        self.termination_counter[i] = 3
                        self.DPP_sol[i]= initial_solution
                        DPP_sol_feasible[i] = 0

            # (3) Check termination criteria
            termination_criteria = self.convergence_checking()
            self.v = self.v + 1
            # Update DPP
            print("update")
            self.update_DPP_set(self.lambda_matrix, self.beta_matrix, self.DPP_sol, DPP_sol_feasible, self.upperbound_matrix)
            # COPY CURRENT LAMBDAS
            for i in range(0,len(self.lambda_matrix)):
                self.lambda_matrix_prev[i] = self.lambda_matrix[i].copy()

        # DESTROY DPP
        self.destroy_DPP_set()
        return self.solution_RPP

###############################################################################
# PRIVATE gather_solution()
###############################################################################
    def gather_solution(self, DPP_sol_row, initial_solution):
        counter = 0
        s = gurobipy.tupledict()
        tr = gurobipy.tupledict()
        r = gurobipy.tupledict()
        er = gurobipy.tupledict()
        e = gurobipy.tupledict()
        Tlen = self.problem_data.get_names("wildfire")
        Ilen = self.problem_data.get_names("resources")
        Glen = self.problem_data.get_names("groups")

        for res in Ilen:
            DPP = DPP_sol_row[counter]
            for tt in Tlen:
                s[res,tt] = DPP.get_variables().get_variable('s')[res,tt].X
                tr[res,tt] = DPP.get_variables().get_variable('tr')[res,tt].X
                r[res,tt] = DPP.get_variables().get_variable('r')[res,tt].X
                er[res,tt] = DPP.get_variables().get_variable('er')[res,tt].X
                e[res,tt] = DPP.get_variables().get_variable('e')[res,tt].X
            counter = counter + 1
        vars = gurobipy.tupledict()
        vars["s"] = s
        vars["tr"] = tr
        vars["r"] = r
        vars["er"] = er
        vars["e"] = e
        modelcopy = initial_solution.get_model().copy()

        counter=0
        for res in Ilen:
            DPP = DPP_sol_row[counter]
            for tt in Tlen:
                modelcopy.getVarByName("start["+res+","+str(tt)+"]").start = DPP.get_variables().get_variable('s')[res,tt].X
                modelcopy.getVarByName("travel["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('tr')[res,tt].X
                modelcopy.getVarByName("rest["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('r')[res,tt].X
                modelcopy.getVarByName("end_rest["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('er')[res,tt].X
                modelcopy.getVarByName("end["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('e')[res,tt].X
            counter = counter + 1
        for gro in Glen:
            for tt in Tlen:
                modelcopy.getVarByName("missing_resources["+gro+","+str(tt)+"]").start = DPP.get_variables().get_variable('mu')[gro,tt].X

        modelcopy.update()
        sol1 = _sol.Solution(modelcopy, vars)

        return sol1

    def create_initial_solution(self, isol, osol):
        Tlen = self.problem_data.get_names("wildfire")
        Ilen = self.problem_data.get_names("resources")
        Glen = self.problem_data.get_names("groups")

        for res in Ilen:
            for tt in Tlen:
                if isinstance(isol, dict):
                    osol.get_model().getVarByName("start["+res+","+str(tt)+"]").start = isol['s'][res,tt]
                    osol.get_model().getVarByName("travel["+str(res)+","+str(tt)+"]").start = isol['tr'][res,tt]
                    osol.get_model().getVarByName("rest["+str(res)+","+str(tt)+"]").start = isol['r'][res,tt]
                    osol.get_model().getVarByName("end_rest["+str(res)+","+str(tt)+"]").start = isol['er'][res,tt]
                    osol.get_model().getVarByName("end["+str(res)+","+str(tt)+"]").start = isol['e'][res,tt]
                elif isinstance(isol, _sol.Solution):
                    print(isol.get_model().getVarByName("start["+res+","+str(tt)+"]").start)
                    osol.get_model().getVarByName("start["+res+","+str(tt)+"]").start = isol.get_variables().get_variable('s')[res,tt].X
                    osol.get_model().getVarByName("travel["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('tr')[res,tt].X
                    osol.get_model().getVarByName("rest["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('r')[res,tt].X
                    osol.get_model().getVarByName("end_rest["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('er')[res,tt].X
                    osol.get_model().getVarByName("end["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('e')[res,tt].X
        for gro in Glen:
            for tt in Tlen:
                if isinstance(isol, dict):
                    osol.get_model().getVarByName("missing_resources["+gro+","+str(tt)+"]").start = isol['mu'][gro,tt]
                elif isinstance(isol, _sol.Solution):
                    osol.get_model().getVarByName("missing_resources["+gro+","+str(tt)+"]").start = isol.get_variables().get_variable('mu')[gro,tt].X

        osol.get_model().update()

        return osol

    def print_solution(self, solution):
        Tlen = self.problem_data.get_names("wildfire")
        Ilen = self.problem_data.get_names("resources")
        for res in Ilen:
            for tt in Tlen:
                print(solution.get_variables().tr[res,tt])

###############################################################################
# PRIVATE create_DPP_set()
###############################################################################
    def create_DPP_set(self):
        for i in range(0,self.y_master_size-1):
            print("Create problem ("+str(i)+")")
            problem_DPP_row = []
            self.y_master = dict([ (p, 1) for p in range(0,self.y_master_size)])
            for p in range(self.y_master_size - (1+i), self.y_master_size):
                self.y_master[p] = 0
            for j in range(0,self.N):
                problem_DPP_row.append(ADPP.DecomposedPrimalProblem(self.problem_data,
                                       self.lambda_matrix[i], self.beta_matrix[i], j,
                                       self.y_master, self.DPP_sol[i],  self.N, self.NL, self.upperbound_matrix[i]))
            self.problem_DPP.append(problem_DPP_row)

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
        for i in range(0,len(self.problem_DPP)):
            len_p = len(self.problem_DPP[i])
            if (len_p > 0):
                for j in range(0,len_p):
                    del self.problem_DPP[i][0]
                self.problem_DPP[i]=[]
        self.problem_DPP = []
        print("DESTROY")

###############################################################################
# PRIVATE destroy_DPP_set()
###############################################################################
    def update_DPP_set(self, lambda_matrix, beta_matrix, solution, DPP_feasible, upperbound_matrix):
        for i in range(0,self.y_master_size-1):
            for j in range(0,self.N):
                if (DPP_feasible[i] == 1):
                    self.problem_DPP[i][j].update_model(lambda_matrix[i], beta_matrix[i], solution[i], upperbound_matrix[i])


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
