"""Module with lagrangian decomposition methods."""
# Python packages

# Package modules
import logging as log
from firedecomp import LR
from firedecomp import original
from firedecomp import logging
from firedecomp.LR import RPP
from firedecomp.LR import DPP
from firedecomp.LR import RDP
import time
import math


###############################################################################
# CLASS LagrangianRelaxation()
###############################################################################
class LagrangianRelaxation(object):
    def __init__(
        self,
        problem_data,
        min_res_penalty=1000000,
        gap=0.01,
        max_iters=100,
        max_time=60,
        log_level="LR",
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
                'MIPGap': 0.0,
                'MIPGapAbs': 0.0,
                'OutputFlag': 0,
                'LogToConsole': 0,
            }
        self.solver_options = solver_options
        # Log level
        self.__log__(log_level)
        # Subgradient vars
        self.a = 1
        self.b = 0.001
        # Create lambda
        self.NL = (1 + len(problem_data.get_names("wildfire")) +
            len(problem_data.get_names("wildfire"))*len(problem_data.get_names("groups"))*2)
        self.lambda1 = []
        self.lambda1_prev = []
        self.lambda1_next = []
        init_value=10
        for i in range(0,self.NL):
            self.lambda1.append(init_value)
            self.lambda1_prev.append(init_value)
            self.lambda1_next.append(init_value)
        # Create Relaxed Primal Problem
        self.problem_RPP = RPP.RelaxedPrimalProblem(problem_data, self.lambda1);
        self.solution_RPP = float("inf")
        # Initialize Decomposite Primal Problem Variables
        self.problem_DPP = []
        self.N = len(self.problem_RPP.I)
        self.groupR = []
        self.y_master_size = self.problem_RPP.return_sizey()


###############################################################################
# PUBLIC METHOD subgradient()
###############################################################################
    def subgradient(self):
        # solution, lambda, mi
        # update lambda and mi
        part1 = 1 / (self.a + self.b/self.v)
        if (self.LR_pen != 0):
            part2 = (self.LR_pen) / abs(self.LR_pen)
        else:
            part2 = 0

        for i in range(0,self.NL):
            self.lambda1_next[i] = self.lambda1[i] + part1 * part2
            if (self.lambda1_next[i] < 0.0):
                self.lambda1_next[i] = self.lambda1[i]
                self.a = self.a*10
                self.b = self.b*10

        print("self.lambda1_next :"+str(self.lambda1_next[0]))
        print("self.lambda1      :"+str(self.lambda1[0]))
        print("self.lambda1_prev :"+str(self.lambda1_prev[0]))

        return self.lambda1_next

###############################################################################
# PUBLIC METHOD convergence_checking()
###############################################################################
    def convergence_checking(self):
        stop = bool(False)
        result = abs(self.lambda1_prev[0]-self.lambda1_next[0])/abs(self.lambda1[0])
        # check convergence
        if (self.v >= self.max_iters):
            print("[STOP] Max iters achieved!")
            stop = bool(True)
        elif (result <= self.epsilon):
            print("[STOP] Lambda epsilion achieved!")
            stop = bool(True)

        return stop

###############################################################################
# PUBLIC METHOD solve()
###############################################################################
    def solve(self, max_iters=100):

        termination_criteria = bool(False)

        while (termination_criteria == False):
            # (0) Initialize DPP
            self.create_DPP_set()
            # (1) Solve DPP problems
            self.DPP_sol = []
            self.L_obj_down = 0
            self.obj = 0
            self.LR_pen = 0
            best_f_value = float("inf")
            self.lambda1 = self.lambda1_next.copy()
            #for i in range(0,self.y_master_size-1):
            #    DPP_sol_row = []
            #    fl_value=0
            #    f_value=0
            #    fp_value=0
            #    for j in range(0,self.N):
            #        DPP_sol_row.append(self.problem_DPP[i][j].solve(self.solver_options))
            #        fl_value = fl_value + DPP_sol_row[j].get_objfunction()
            #        f_value  = f_value + self.problem_DPP[i][j].return_function_obj(DPP_sol_row[j])
            #        fp_value = fp_value + self.problem_DPP[i][j].return_LR_obj(DPP_sol_row[j])
            #    if ((f_value > 0) == (best_f_value > f_value)):
            #        self.L_obj_down  = fl_value
            #        self.obj = f_value
            #        self.LR_pen = fp_value
            #        best_f_value = f_value
            #    self.DPP_sol.append(DPP_sol_row)
            self.problem_RPP = RPP.RelaxedPrimalProblem(self.problem_data, self.lambda1);
            self.solution_RPP = self.problem_RPP.solve(self.solver_options)
            self.L_obj_down = self.solution_RPP.get_objfunction()
            self.obj = self.problem_RPP.return_function_obj(self.solution_RPP)
            self.LR_pen = self.problem_RPP.return_LR_obj2(self.solution_RPP)
            self.pen_all = self.problem_RPP.return_LR_obj(self.solution_RPP)
            print("self.L_obj_down "+str(self.L_obj_down))
            print("self.obj "+str(self.obj))
            print("self.LR_pen "+str(self.LR_pen))
            print("self.pen_all "+str(self.pen_all))
            print("self.lambda "+str(self.lambda1))

            # (2) Calculate new values of lambda and update
            self.lambda1_prev = self.lambda1.copy()
            self.lambda1_next = self.subgradient()

            if (self.L_obj_down > self.L_obj_down_prev):
                self.L_obj_down_prev = self.L_obj_down

            # Show iteration results
            #self.problem_data.original_model.insert_soludion(self.DPP_sol,
            #            self.option_decomp, self.lambda1, self.solver_options,
            #            self.groupR)
            log.info("Iteration # mi lambda f(x) L(x,mi,lambda) penL")
            print("Iter: "+str(self.v)+ " Lambda: "+str(self.lambda1[0])+
                    " LR(x): "+str(self.L_obj_down)+" f(x):"+ str(self.obj) +
                        " penL:" + str(self.LR_pen) +"\n")



            # (3) Check termination criteria
            termination_criteria = self.convergence_checking()
            if termination_criteria == False:
                self.destroy_DPP_set()
            self.v = self.v + 1

        return self.solution_RPP

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
                problem_DPP_row.append(DPP.DecomposedPrimalProblem(self.problem_data,
                                                    self.lambda1, j, self.y_master))
                self.problem_DPP.append(problem_DPP_row)

###############################################################################
# PRIVATE destroy_DPP_set()
###############################################################################
    def destroy_DPP_set(self):
        self.problem_DPP = []


###############################################################################
# PRIVATE METHOD __log__()
###############################################################################
    def __log__(self, level="LR"):
        log.addLevelName(80, "LR")
        log.Logger.LR = logging.LR

        if level != 'LR':
            log_level = getattr(log, level)
            logger = log.getLogger('LR_logging')
            logger.setLevel(log_level)
            logger.addFilter(logging.LRFilter())
            if len(logger.handlers) == 0:
                ch = log.StreamHandler()
                ch.setLevel(log_level)
                # create formatter and add it to the handlers
                formatter = log.Formatter("%(levelname)8s: %(message)s")
                ch.setFormatter(formatter)
                logger.addHandler(ch)
        else:
            log_level = 80
            logger = log.getLogger('LR')
            logger.setLevel(log_level)
            if len(logger.handlers) == 0:
                ch = log.StreamHandler()
                ch.setLevel(log_level)
                # create formatter and add it to the handlers
                formatter = log.Formatter("%(message)s")
                ch.setFormatter(formatter)
                logger.addHandler(ch)

        self.log = logger
        return 1
