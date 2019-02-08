"""Module with lagrangian decomposition methods."""
# Python packages

# Package modules
import logging as log
from firedecomp import LR
from firedecomp import original
from firedecomp import logging
from firedecomp.LR import RPP
from firedecomp.LR import DPP
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
        self.L_obj_down = float("-inf")
        self.gap = float("inf")
        # OPTIONS LR
        self.max_iters = max_iters
        self.max_time = max_time
        self.v = 1
        self.lambda1 = [1.0, 1.0, 1.0, 1.0]
        self.lambda1_prev = [0.0, 0.0, 0.0, 0.0]
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
        # Create Relaxed Primal Problem
        self.problem_RPP = RPP.RelaxedPrimalProblem(problem_data, self.lambda1);
        self.RPP_sol = float("inf")
        # Create Decomposite Primal Problem
        if (option_decomp == 'R'):
            self.NR = problem_data.get_resources().size()
        elif (option_decomp == 'G'):
            self.NR = len(problem_data.get_names("groups"))
        else:
            self.NR = problem_data.get_resources().size()
        self.problem_DPP = []
        self.DPP_sol = []
        for i in range(0,self.NR):
            self.problem_DPP.append(DPP.DecomposedPrimalProblem(problem_data,
                                            self.lambda1, i, option_decomp))
            self.DPP_sol.append(float("inf"))

###############################################################################
# PUBLIC METHOD subgradient()
###############################################################################
    def subgradient(self):
        # solution, lambda, mi
        # update lambda and mi
        num = 0
        aux = 0
        for i in range(0,self.NR):
            num = num + (-1) *  self.DPP_sol[i]
        for i in range(0,self.NR):
            aux = aux + self.DPP_sol[i]**2
        module = math.sqrt(aux)
        l = (self.lambda1
                        + (1/(self.a+self.b*self.v))
                        * num / module)
        for i in range(0,self.NR):
            self.lambda1[i] = l
        return self.lambda1

###############################################################################
# PUBLIC METHOD convergence_checking()
###############################################################################
    def convergence_checking(self, max_iters):
        stop = bool(False)
        # check convergence

        if (self.v >= max_iters):
            stop = bool(True)

        return stop

###############################################################################
# PUBLIC METHOD solve()
###############################################################################
    def solve(self, max_iters=10):

        termination_criteria = bool(False)


        while (termination_criteria == False):
            # (1) Solve DPP problems
            for i in range(0,self.NR):
                model = self.problem_DPP[i].solve(self.solver_options)
                self.DPP_sol[i] = model.get_objfunction()
                #print(self.DPP_sol[i])
                #print("\n\n")
                #print(model.get_variables().get_names())
                #print(model.get_variables().get_variables().get_element('s'))
                #print("\n\n")
            # (2) Calculate new values of lambda and update
            self.lambda1_prev = self.lambda1
            self.subgradient()
            # (3) Check termination criteria
            termination_criteria = convergence_checking(max_iters)
            self.v = self.v + 1

            # Update lambda in RPP and DPP
            self.problem_RPP.update_lambda1(self.lambda1)
            for i in range(0,self.NR):
                self.problem_DPP[i].update_lambda1(self.lambda1)
            # Update solution in RPP and original model
            self.L_obj_up = self.__set_solution_in_RPP__()
            if (self.L_obj_up > self.L_obj_down):
                self.L_obj_down = self.L_obj_up 
            self.__set_solution_in_original_model__()
            # Show iteration results
            log.info("Iteration # mi lambda f(x) L(x,mi,lambda)")

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
# PRIVATE METHOD __set_solution_in_RPP__()
###############################################################################
    def __set_solution_in_RPP__(self, solutions):

        return 1

###############################################################################
# PRIVATE METHOD __log__()
###############################################################################
    def __set_solution_in_original_model__(self, solutions):

        return 1
