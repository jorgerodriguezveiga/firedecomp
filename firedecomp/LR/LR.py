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
        max_iters=10,
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
        self.solver_options = solver_options
        # Chargue models
        self.current_solution = []
        # Log level
        self.__log__(log_level)
        # subgradient
        self.a = 1
        self.b = 0.1
        # Create lambda
        self.NL = 4
        self.lambda1 = []
        self.lambda1_prev = []
        for i in range(0,self.NL):
            self.lambda1.append(1.0)
            self.lambda1_prev.append(0.0)
        # Create Relaxed Primal Problem
        self.problem_RPP = RPP.RelaxedPrimalProblem(problem_data, self.lambda1);
        self.RPP_sol = float("inf")
        # Create Decomposite Primal Problem
        if (self.option_decomp == 'R'):
            self.N = len(self.problem_RPP.I)
            self.groupR = []
        elif (self.option_decomp == 'G'):
            self.N = len(self.problem_RPP.G)
        else:
            print("Error[0] Use 'G' to divide by groups, or 'R' to divide by resources ")
            sys.exit()
        self.problem_DPP = []
        self.DPP_sol_obj = []
        for i in range(0,self.N):
            string_i = self.problem_data.get_names("resources")[i]
            self.problem_DPP.append(DPP.DecomposedPrimalProblem(problem_data,
                                self.lambda1, i, string_i, self.option_decomp))
            self.DPP_sol_obj.append(float("inf"))
            if (self.option_decomp == 'R'):
                self.groupR.append(self.problem_DPP[i].G[0])


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

        self.lambda1[0] = self.lambda1[0] + 0.1
        self.lambda1[1] = self.lambda1[1] + 0.1
        self.lambda1[2] = self.lambda1[2] + 0.1
        self.lambda1[3] = self.lambda1[3] + 0.1

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
    def solve(self, max_iters=10):

        termination_criteria = bool(False)

        while (termination_criteria == False):
            # (1) Solve DPP problems
            self.DPP_sol = []
            for i in range(0,self.N):
                self.DPP_sol.append(self.problem_DPP[i].solve(self.solver_options))
                self.DPP_sol_obj[i] = self.DPP_sol[i].get_objfunction()
            # (2) Calculate new values of lambda and update
            self.lambda1_prev = self.lambda1.copy()
            self.subgradient()
            # (3) Check termination criteria
            termination_criteria = self.convergence_checking()
            self.v = self.v + 1

            # Update solution in RPP.solution problem
            self.problem_RPP.__set_solution_in_RPP__(self.DPP_sol,
                        self.option_decomp, self.lambda1, self.solver_options,
                        self.groupR)
            self.RPP_sol = self.problem_RPP.solve(self.solver_options)
            self.L_obj_down = self.RPP_sol.get_objfunction()
            if (self.L_obj_down > self.L_obj_down_prev):
                self.L_obj_down_prev = self.L_obj_down

            # Extract Upper Bound


            # Dual gap
            #RDP.RelaxedDualProblem(self.problem_data, self.lambda1,
            #            primal=self.problem_RPP, solution=self.RPP_sol,
            #            option_decomp=self.option_decomp);

            # Update lambda in RPP and DPP
            self.problem_RPP.update_lambda1(self.lambda1)
            for i in range(0,self.N):
                self.problem_DPP[i].update_lambda1(self.lambda1)

            # Show iteration results
            log.info("Iteration # mi lambda f(x) L(x,mi,lambda)")
            print("Iter: "+str(self.v)+ " Lambda: "+str(self.lambda1_prev)+" LowerBound: "+str(self.L_obj_down)+" Duality Gap\n")

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

###############################################################################
# PRIVATE METHOD __log__()
###############################################################################
    def __set_solution_in_original_model__(self, solutions):

        return 1
