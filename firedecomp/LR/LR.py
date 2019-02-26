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
        option_decomp='G',
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
        self.v = 1
        self.RPP_obj_prev = float("inf")
        self.option_decomp = option_decomp
        # Gurobi options
        if solver_options == None:
            solver_options = {
                #'MIPGap': 0.0,
                #'MIPGapAbs': 0.0,
                'OutputFlag': 0,
                'LogToConsole': 0,
            }
        self.solver_options = solver_options
        # Log level
        self.__log__(log_level)
        # Subgradient vars
        self.a = 1
        self.b = 0.1
        # Create lambda
        self.NL = 4
        self.lambda1 = []
        self.lambda1_prev = []
        for i in range(0,self.NL):
            self.lambda1.append(0.1)
            self.lambda1_prev.append(0.0)
        # Create Relaxed Primal Problem
        self.problem_RPP = RPP.RelaxedPrimalProblem(problem_data, self.lambda1);
        self.RPP_sol = float("inf")
        # Initialize Decomposite Primal Problem Variables
        self.N = len(self.problem_RPP.I)
        self.groupR = []
        self.y_master_size = self.problem_RPP.return_sizey()


###############################################################################
# PUBLIC METHOD subgradient()
###############################################################################
    def subgradient(self):
        # solution, lambda, mi
        # update lambda and mi
        #num = 0
        #aux = 0
        #for i in range(0,self.N):
        #    num = num + (-1) *  self.DPP_sol_obj[i]
        #for i in range(0,self.N):
        #    aux = aux + (-1) * self.DPP_sol_obj[i]**2
        #module = math.sqrt(aux)
        #l = (1/(self.a+self.b*self.v)) * (num / module)
        #for i in range(0,self.NL):
        #    self.lambda1[i] = self.lambda1[i] + l

        const=1
        self.lambda1[0] = self.lambda1[0] + const
        self.lambda1[1] = self.lambda1[1] + const
        self.lambda1[2] = self.lambda1[2] + const
        self.lambda1[3] = self.lambda1[3] + const

        return self.lambda1

###############################################################################
# PUBLIC METHOD convergence_checking()
###############################################################################
    def convergence_checking(self):
        stop = bool(False)
        # check convergence
        if (self.v >= self.max_iters):
            stop = bool(True)

        return stop

###############################################################################
# PUBLIC METHOD solve()
###############################################################################
    def solve(self, max_iters=100):

        termination_criteria = bool(False)

        while (termination_criteria == False):
            # (0) Initialize DPP
            self.problem_DPP = []
            self.DPP_sol_obj = []
            for i in range(0,self.N):
                problem_DPP_row = []
                for j in range(0,self.y_master_size-1):
                    self.y_master = dict([ (p, 1) for p in range(0,self.y_master_size)])
                    for p in range(self.y_master_size - (1+j), self.y_master_size):
                        self.y_master[p] = 0
                    print(self.y_master)
                    problem_DPP_row.append(DPP.DecomposedPrimalProblem(self.problem_data,
                                    self.lambda1, i, self.y_master))
                self.problem_DPP.append(problem_DPP_row)
            # (1) Solve DPP problems
            self.DPP_sol = []
            self.L_obj_down = 0
            self.obj = 0
            self.LR_pen = 0
            for i in range(0,self.N):
                DPP_sol_row = []
                for j in range(0,self.y_master_size-1):
                    DPP_sol_row.append(self.problem_DPP[i][j].solve(self.solver_options))
                    fl_value = DPP_sol_row[j].get_objfunction()
                    time.sleep(10)
                    f_value  = self.problem_DPP[i][j].return_function_obj(DPP_sol_row[j])
                    fp_value = self.problem_DPP[i][j].return_LR_obj(DPP_sol_row[j])
                    self.L_obj_down  = self.L_obj_down  + fl_value
                    self.obj = self.obj + f_value
                    self.LR_pen = self.LR_pen + fp_value
                self.DPP_sol.append(DPP_sol_row)

            # (2) Calculate new values of lambda and update
            self.lambda1_prev = self.lambda1.copy()
            self.subgradient()
            # (3) Check termination criteria
            termination_criteria = self.convergence_checking()
            self.v = self.v + 1

            # Update solution in RPP.solution problem
            #self.problem_RPP.__set_solution_in_RPP__(self.DPP_sol,
            #            self.option_decomp, self.lambda1, self.solver_options,
            #            self.groupR)
            #self.RPP_sol = self.problem_RPP.solve(self.solver_options)
            if (self.L_obj_down > self.L_obj_down_prev):
                self.L_obj_down_prev = self.L_obj_down

            # Extract Upper Bound
            # ToDo

            # Dual gap
            #RDP.RelaxedDualProblem(self.problem_data, self.lambda1,
            #            primal=self.problem_RPP, solution=self.RPP_sol,
            #            option_decomp=self.option_decomp);

            # Update lambda in RPP and DPP
            self.problem_RPP.update_lambda1(self.lambda1)
            for i in range(0,self.N):
                self.problem_DPP[i].update_lambda1(self.lambda1)


            # Show iteration results
            #self.problem_data.original_model.insert_soludion(self.DPP_sol,
            #            self.option_decomp, self.lambda1, self.solver_options,
            #            self.groupR)
            log.info("Iteration # mi lambda f(x) L(x,mi,lambda) penL")
            print("Iter: "+str(self.v)+ " Lambda: "+str(self.lambda1_prev)+
                    " LR(x): "+str(self.L_obj_down)+" f(x):"+ str(self.obj) +
                        " penL:" + str(self.LR_pen) +"\n")

            #sleep(10)

        return 1

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
