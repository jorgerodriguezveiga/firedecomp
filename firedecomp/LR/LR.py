# Python packages
import logging as log

# Package modules
from firedecomp import LR
from firedecomp import original


# Benders ---------------------------------------------------------------------
class LagrangianRelaxation(object):
    def __init__(self, problem_data, min_res_penalty=1000000, gap=0.01):
        if problem_data.period_unit is False:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data

        # Initialization LR variables 
        self.lb = -float("-inf")
        self.ub = float("inf")
        self.v = 1
	self.lambda = 1
	self.mi = 1

        # Chargue models
        self.problem = original.model.InputModel(
            problem_data, relaxed=False, min_res_penalty=min_res_penalty)
        
	self.RPP = LR.RPP.RelaxedPrimalProblem(
            problem_data, self.problem)

	self.DPP_list = []
        self.DPP_list = self.RPP.return_DPP()

        self.solution = []
	self.fitness = float(inf)
	self.Lfitness = float(inf)
        self.iter = 1

    def subgradient(self):
        # solution, lambda, mi
        # update lambda and mi
	

    def convergence_checking(self, max_iters):
        bool stop = False
        # check convergence

        if (self.iter >= max_iters):
            stop = True

        return stop


    def solveLR(self, max_iters=10):

        bool termination_criteria = False

        while (termination_criteria == False):
            for problem_DLP in  DLP_list:
                problem_DLP.solve()
                problem_DLP.update_solution(self.solution)

            self.Lfitness = problem_RPP.evaluate(self.solution,self.lambda,self.mi)
            self.Lfitness = self.problem
            subgradient()
	    termination_criteria = convergence_checking(max_iters)
            self.iter = self.iter + 1
            log.info("Iteration # mi lambda f(x) L(x,mi,lambda)")
            
# --------------------------------------------------------------------------- #
