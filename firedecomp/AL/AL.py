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
        # OBJECTIVE FUNCTION
        self.obj = float("inf")
        self.L_obj_up = float("inf")
        self.L_obj_up_prev = float("inf")
        self.L_obj_down = float("-inf")
        self.L_obj_down_prev = float("-inf")
        self.gap = float("inf")
        # OPTIONS LR
        self.max_iters = max_iters
        self.max_time = max_time
        self.epsilon = 0.000001
        self.v = 1
        self.RPP_obj_prev = float("inf")
        # Gurobi options
        if solver_options == None:
            solver_options = {
                #'MIPGap': 0.01,
                #'MIPGapAbs': 0.01,
                'OutputFlag': 0,
                'LogToConsole': 0,
            }
        self.solver_options = solver_options
        # Log level
        #self.__log__(log_level)
        # Create lambda
        #self.NL = (1 + len(problem_data.get_names("wildfire")) +
        #        len(problem_data.get_names("wildfire"))*len(problem_data.get_names("groups"))*2)

        self.NL =  1 + len(problem_data.get_names("wildfire"))
        self.lambda1 = []
        self.lambda1_change = []
        self.beta = []
        self.lambda1_prev = []
        self.lambda1_next = []
        self.lambda_matrix = []
        self.beta_matrix = []
        self.L_obj_down  = -1e16
        self.obj = float("inf")
        self.LR_pen = []
        self.inf_sol = float("inf")
        self.pen_all = float("inf")
        self.index_best = -1
        init_value=1e3
        for i in range(0,self.NL):
            self.lambda1.append(init_value)
            self.lambda1_change.append(init_value)
            self.beta.append(0.5)
            self.LR_pen.append(0)
        # Create Relaxed Primal Problem
        self.problem_RPP = ARPP.RelaxedPrimalProblem(problem_data, self.lambda1, self.beta);
        self.solution_RPP = float("inf")
        # Initialize Decomposite Primal Problem Variables
        self.problem_DPP = []
        self.N = len(self.problem_RPP.I)
        self.groupR = []
        self.y_master_size = self.problem_RPP.return_sizey()
        self.counterh_matrix = []
        init_value=100
        for i in range(0,self.y_master_size):
            lambda_row = []
            beta_row = []
            counterh_row = []
            for j in range(0,self.NL):
                lambda_row.append(init_value+j)
                beta_row.append(0.5)
                counterh_row.append(0)
            self.lambda_matrix.append(lambda_row)
            self.beta_matrix.append(beta_row)
            self.counterh_matrix.append(counterh_row)


###############################################################################
# PUBLIC METHOD subgradient()
###############################################################################
    def subgradient(self, LR_pen_v, lambda_vector, beta_vector, ii):

        lambda_old = lambda_vector.copy()
        beta_old = beta_vector.copy()

        for i in range(0,self.NL):

            LRpen = LR_pen_v[i]

            lambda_vector[i] = max ( 0, lambda_old[i] + beta_old[i] * LRpen )
            if abs(lambda_vector[i]) > 0:
                change_per = abs(abs(lambda_vector[i])-abs(self.lambda1_change[i]))
                #change_per = abs(abs(lambda_vector[i])-abs(self.lambda1_change[i]))/abs(lambda_vector[i])
            else:
                change_per = 0
            if change_per >= 1 and beta_vector[i] <= 5:
                #self.lambda1_change[i] = lambda_vector[i]
                beta_vector[i] = beta_vector[i] * 1.2

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
        for i in range(0,len(self.lambda1)):
            rr = abs(self.lambda1_prev[i]-self.lambda1_next[i])/abs(self.lambda1[i]+self.epsilon)
            if (rr <= self.epsilon) :
                result = result + 1
        # check convergence
        if (self.v >= self.max_iters):
            print("[STOP] Max iters achieved!")
            stop = bool(True)
        elif (result >=  len(self.lambda1)):
            print("[STOP] Lambda epsilion achieved!")
            stop = bool(True)

        return stop

###############################################################################
# PUBLIC METHOD solve()
###############################################################################
    def solve(self, max_iters=100):

        termination_criteria = bool(False)
        changes = 0

        # (0) Initial solution
        self.DPP_sol = []

        init_options = {
                'IterationLimit': 1000,
                'OutputFlag': 0,
                'LogToConsole': 0,
        }
        #isol = self.problem_data.solve(solver_options=init_options)
        isol = self.problem_RPP.solve(solver_options=init_options)
        initial_solution = self.create_initial_solution(isol)
        DPP_sol_feasible = []
        for i in range(0,self.y_master_size-1):
            DPP_sol_feasible.append(1)
            self.DPP_sol.append(initial_solution)

        # (0) Initialize DPP
        self.create_DPP_set()
        while (termination_criteria == False):
            # (1) Solve DPP problems
            print("ITERATION -> "+str(self.v))
            for i in range(0,self.y_master_size-1):
                print("### START Y -> "+str(self.problem_DPP[i][0].list_y))
                DPP_sol_row = []
                DPP_sol_row_feasible = []
                L_obj_down_local=0

                LR_pen_local = []
                for j in range(0,self.NL):
                    LR_pen_local.append(0.0)
                obj_local=0
                pen_all_local=0
                stop_inf = False
                inf_sol=0
                for j in range(0,self.N):
                    #print("SOLVE "+str(j))
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
                    L_obj_down_local = L_obj_down_local + DPP_sol_row[j].get_objfunction()
                    obj_local = obj_local + self.problem_DPP[i][j].return_function_obj(DPP_sol_row[j])
                    LR_pen_local = self.problem_DPP[i][j].return_LR_obj2(DPP_sol_row[j])
                    pen_all_local = self.problem_DPP[i][j].return_LR_obj(DPP_sol_row[j])
                    inf_sol = self.extract_infeasibility(LR_pen_local)
                    print("XResource"+str(j)+" "+str(i)+
                        " UB "+str(DPP_sol_row[j].get_objfunction())+
                        " fobj "+str(self.problem_DPP[i][j].return_function_obj(DPP_sol_row[j]))+
                        " Infeas "+str(inf_sol) + "  ||  " + str(self.problem_DPP[i][j].return_LR_obj2(DPP_sol_row[j])))
                # update lambda
                for j in range(0,self.N):
                    if (DPP_sol_row_feasible[j] == 0):
                        stop_inf = True
                        break
                    UB_local = DPP_sol_row[j].get_objfunction()
                    LR_pen_local = self.problem_DPP[i][j].return_LR_obj2(DPP_sol_row[j])

                if (self.L_obj_down < L_obj_down_local and (inf_sol <= 0) and (DPP_sol_row_feasible[j] != 0)):
                    self.L_obj_down  = L_obj_down_local
                    self.obj = obj_local
                    self.LR_pen = LR_pen_local
                    self.inf_sol = inf_sol
                    self.pen_all = pen_all_local
                    self.index_best_i = i
                    self.index_best_j = j
                    change=1

                if (stop_inf == False):
                    print("")
                    print("")
                    print("TOTAL = UB "+str(L_obj_down_local)+" fobj "+str(obj_local))
                    print("")
                    print("")

                    print("subgradient")
                    self.subgradient( LR_pen_local, self.lambda_matrix[i], self.beta_matrix[i], i)

                    if (inf_sol <= 0):
                        self.DPP_sol[i]=self.gather_solution(DPP_sol_row, initial_solution)
                    # (2) Calculate new values of lambda and update
                    #if (changes != 0):
                    #    self.lambda1_prev = self.lambda1.copy()
                    #    self.lambda1_next = self.lambda_matrix[self.index_best_i]

                    #if (self.L_obj_down > self.L_obj_down_prev):
                    #    self.L_obj_down_prev = self.L_obj_down

                    # Show iteration results
                    #if i == 1:
                    log.info("Iteration # mi lambda f(x) L(x,mi,lambda) penL")
                    print("Iter: "+str(self.v)+ " Lambda: "+str(self.lambda1[0])+
                            " LR(x): "+str(self.L_obj_down)+" f(x):"+ str(self.obj) +
                            " penL:" + str(self.LR_pen) +"\n")
                else:
                    self.DPP_sol[i]= initial_solution
                    DPP_sol_feasible[i] = 0

            # (3) Check termination criteria
            ##termination_criteria = self.convergence_checking()
            self.v = self.v + 1
            # Update DPP
            self.update_DPP_set(self.lambda_matrix, self.beta_matrix, self.DPP_sol, DPP_sol_feasible)

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
        modelcopy.update()
        sol1 = _sol.Solution(modelcopy, vars)
        counter=0
        for res in Ilen:
            DPP = DPP_sol_row[counter]
            for tt in Tlen:
                sol1.get_model().getVarByName("start["+res+","+str(tt)+"]").start = DPP.get_variables().get_variable('s')[res,tt].X
                sol1.get_model().getVarByName("travel["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('tr')[res,tt].X
                sol1.get_model().getVarByName("rest["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('r')[res,tt].X
                sol1.get_model().getVarByName("end_rest["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('er')[res,tt].X
                sol1.get_model().getVarByName("end["+str(res)+","+str(tt)+"]").start = DPP.get_variables().get_variable('e')[res,tt].X
            counter = counter + 1
        #for gro in Glen:
        #    for tt in Tlen:
        #        sol1.get_model().getVarByName("missing_resources["+gro+","+str(tt)+"]").start = DPP.get_variables().get_variable('mu')[gro,tt].X
        sol1.get_model().update()
        #print(sol1.get_model().getVars())
        #   mu[res,tt] = DPP.get_variables().get_variable('mu')[res,tt]
        return sol1

    def create_initial_solution(self, isol):
        Tlen = self.problem_data.get_names("wildfire")
        Ilen = self.problem_data.get_names("resources")
        Glen = self.problem_data.get_names("groups")

        for res in Ilen:
            for tt in Tlen:
                isol.get_model().getVarByName("start["+res+","+str(tt)+"]")
                isol.get_model().getVarByName("start["+res+","+str(tt)+"]").start = isol.get_variables().get_variable('s')[res,tt].X
                isol.get_model().getVarByName("travel["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('tr')[res,tt].X
                isol.get_model().getVarByName("rest["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('r')[res,tt].X
                isol.get_model().getVarByName("end_rest["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('er')[res,tt].X
                isol.get_model().getVarByName("end["+str(res)+","+str(tt)+"]").start = isol.get_variables().get_variable('e')[res,tt].X
        for gro in Glen:
            for tt in Tlen:
                isol.get_model().getVarByName("missing_resources["+gro+","+str(tt)+"]").start = isol.get_variables().get_variable('mu')[gro,tt].X
        isol.get_model().update()

        return isol

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
            problem_DPP_row = []
            self.y_master = dict([ (p, 1) for p in range(0,self.y_master_size)])
            for p in range(self.y_master_size - (1+i), self.y_master_size):
                self.y_master[p] = 0
            for j in range(0,self.N):
                problem_DPP_row.append(ADPP.DecomposedPrimalProblem(self.problem_data,
                                       self.lambda_matrix[i], self.beta_matrix[i], j,
                                       self.y_master, self.DPP_sol[i],  self.N, self.NL))
            self.problem_DPP.append(problem_DPP_row)

###############################################################################
# PRIVATE extract_infeasibility()
###############################################################################
    def extract_infeasibility(self, LR_pen_local):
        infeas = 0
        for i in range(0, len(LR_pen_local)):
            if (LR_pen_local[i] > 0):
                infeas = infeas + LR_pen_local[i]
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
    def update_DPP_set(self, lambda_matrix, beta_matrix, solution, DPP_feasible):
        for i in range(0,self.y_master_size-1):
            for j in range(0,self.N):
                if (DPP_feasible[i] == 1):
                    self.problem_DPP[i][j].update_model(lambda_matrix[i], beta_matrix[i], solution[i])


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
