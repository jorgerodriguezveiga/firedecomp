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
from firedecomp.classes import solution as _sol
import time
import math
import gurobipy
import copy

###############################################################################
# CLASS LagrangianRelaxation()
###############################################################################
class LagrangianRelaxation(object):
    def __init__(
        self,
        problem_data,
        min_res_penalty=1000000,
        gap=0.01,
        max_iters=100000,
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
        self.a = 0.1
        self.b = 0.1
        # Create lambda
        #self.NL = 1
        self.NL = (1 + len(problem_data.get_names("wildfire"))  +
            len(problem_data.get_names("wildfire"))*len(problem_data.get_names("groups"))*2)
        #self.NL = (len(problem_data.get_names("wildfire"))  +
        #    len(problem_data.get_names("wildfire"))*len(problem_data.get_names("groups"))*2)
        self.lambda1 = []
        self.lambda1_prev = []
        self.lambda1_next = []
        self.lambda_matrix = []
        self.L_obj_down  = -1e16
        self.obj = float("inf")
        self.LR_pen = []
        self.inf_sol = float("inf")
        self.pen_all = float("inf")
        self.index_best = -1
        self.his = []

        self.UB=2500
        init_value=2
        for i in range(0,self.NL):
            self.lambda1.append(init_value)
            self.lambda1_prev.append(init_value+1)
            self.lambda1_next.append(init_value-1)
            self.his.append(0.0)
        # Create Relaxed Primal Problem
        self.problem_RPP = RPP.RelaxedPrimalProblem(problem_data, self.lambda1);
        self.solution_RPP = float("inf")
        # Initialize Decomposite Primal Problem Variables
        self.problem_DPP = []
        self.N = len(self.problem_RPP.I)
        self.groupR = []
        self.y_master_size = self.problem_RPP.return_sizey()
        for i in range(0,self.y_master_size):
            lambda_row = []
            for j in range(0,self.N):
                lambda_col = []
                for z in range(0,self.NL):
                    lambda_col.append(init_value*(j+1))
                lambda_row.append(lambda_col)
            self.lambda_matrix.append(lambda_row)


###############################################################################
# PUBLIC METHOD subgradient()
###############################################################################
    def subgradient(self, lambda_vector, L_obj_down, LR_pen_v):
        # solution, lambda, mi
        # update lambda and mi
        #feas = all([val <= 0 for val in self.LR_pen])
        total_LRpen = sum([LR_pen_v[i]**2 for i in range(0,self.NL)])
        lambda_old = lambda_vector.copy()
        for i in range(0,self.NL):
            LRpen = LR_pen_v[i]
            part1 = (self.UB - L_obj_down) / (self.a + self.b*self.v)
            #part2 = 0
            if (total_LRpen != 0):
                part2 =  LRpen / total_LRpen
            #part1 = 1 / (self.a + self.b*self.v)
            #part2 = 0
            #if (LRpen != 0):
            #    part2 =  LRpen
                #part2 = LRpen / abs(total_LRpen)
            lambda_vector[i] = lambda_old[i] + part1 * part2
            if (lambda_vector[i] < 0.0):
                lambda_vector[i] = 0

            print("#####"+str(LRpen)+" ->"+str(lambda_old[i])+ " + "+str(part1 * part2) + " = " + str(lambda_vector[i]) )
        #print("")
        #print("")
        return lambda_vector

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

        print("CREATING STARTING POINT")
        # (0) Initial solution
        self.DPP_sol = []
        #isol = self.problem_data.solve()
        isol = self.problem_RPP.solve()
        initial_solution = self.create_initial_solution(isol)
        for i in range(0,self.y_master_size-1):
            self.DPP_sol.append(initial_solution)
        print("START SUBGRADIENT METHOD")
        while (termination_criteria == False):
            # (0) Initialize DPP
            #for i in range(0,self.y_master_size-1):
            #    print("CHECK "+str(i))
            #    self.print_solution(self.DPP_sol[i])

            self.create_DPP_set()
            self.DPP_sol = []
            if (changes != 0):
                self.lambda1 = self.lambda1_next.copy()
                changes=0

            # (1) Solve DPP problems
            print("ITERATION -> "+str(self.v))
            for i in range(0,self.y_master_size-1):

                print("START Y -> "+str(self.problem_DPP[i][0].list_y))

                DPP_sol_row = []
                L_obj_down_local=0

                LR_pen_local = []
                for j in range(0,self.NL):
                    LR_pen_local.append(0.0)
                obj_local=0
                pen_all_local=0
                stop_inf = False
                for j in range(0,self.N):
                    if (i > (self.y_master_size-1)/2):
                        stop_inf = True
                        break
                    #print("SOLVE "+str(j))
                    DPP_sol_row.append(self.problem_DPP[i][j].solve(self.solver_options))
                    if (DPP_sol_row[j].model.Status == 3):
                    #    print("UNFACTIBLE Y"+str(i))
                        stop_inf = True
                        break
                    L_obj_down_local = DPP_sol_row[j].get_objfunction()
                    obj_local  = self.problem_DPP[i][j].return_function_obj(DPP_sol_row[j])
                    LR_pen_local = self.problem_DPP[i][j].return_LR_obj2(DPP_sol_row[j])
                    pen_all_local = self.problem_DPP[i][j].return_LR_obj(DPP_sol_row[j])
                    self.lambda_matrix[i][j] = self.subgradient( self.lambda_matrix[i][j], L_obj_down_local, LR_pen_local)
                    inf_sol = self.extract_infeasibility(LR_pen_local)
                    print(str(j)+" "+str(i)+" UB "+str(L_obj_down_local)+" fobj "+str(obj_local)+" Infeas "+str(inf_sol))
                    if (self.obj >= obj_local and (inf_sol <= 0)):
                        if (self.L_obj_down <= L_obj_down_local):
                            self.L_obj_down  = L_obj_down_local
                            self.obj = obj_local
                            self.LR_pen = LR_pen_local.copy()
                            self.inf_sol = inf_sol
                            self.pen_all = pen_all_local
                            self.index_best_i = i
                            self.index_best_j = j
                            changes=1
                if (stop_inf == False):
                    print("BEST" + str(self.L_obj_down) )
                    #print("")
                    #print("")
                    self.DPP_sol.append(self.gather_solution(DPP_sol_row, initial_solution))
                    # (2) Calculate new values of lambda and update
                    if (changes != 0):
                        self.lambda1_prev = self.lambda1.copy()
                        self.lambda1_next = self.lambda_matrix[self.index_best_i][self.index_best_j]

                    if (self.L_obj_down > self.L_obj_down_prev):
                        self.L_obj_down_prev = self.L_obj_down

                    # Show iteration results
                    log.info("Iteration # mi lambda f(x) L(x,mi,lambda) penL")
                    print("Iter: "+str(self.v)+ " Lambda: "+str(self.lambda1[0])+
                            " LR(x): "+str(self.L_obj_down)+" f(x):"+ str(self.obj) +
                            " penL:" + str(sum(self.LR_pen)) +"\n")
                else:
                    self.DPP_sol.append( initial_solution )

            # (3) Check termination criteria
            termination_criteria = self.convergence_checking()
            self.v = self.v + 1

            # Destroy DPP
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
        sol1.get_model().update()
        #print(sol1.get_model().getVars())
        #   mu[res,tt] = DPP.get_variables().get_variable('mu')[res,tt]
        return sol1

    def create_initial_solution(self, solution):
        Tlen = self.problem_data.get_names("wildfire")
        Ilen = self.problem_data.get_names("resources")
        for res in Ilen:
            for tt in Tlen:
                solution.get_model().getVarByName("start["+res+","+str(tt)+"]").start = solution.get_variables().get_variable('s')[res,tt].X
                solution.get_model().getVarByName("travel["+str(res)+","+str(tt)+"]").start = solution.get_variables().get_variable('tr')[res,tt].X
                solution.get_model().getVarByName("rest["+str(res)+","+str(tt)+"]").start = solution.get_variables().get_variable('r')[res,tt].X
                solution.get_model().getVarByName("end_rest["+str(res)+","+str(tt)+"]").start = solution.get_variables().get_variable('er')[res,tt].X
                solution.get_model().getVarByName("end["+str(res)+","+str(tt)+"]").start = solution.get_variables().get_variable('e')[res,tt].X
        solution.get_model().update()

        #   mu[res,tt] = DPP.get_variables().get_variable('mu')[res,tt]
        return solution


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
                #print("CREATE DPP "+str(j))
                problem_DPP_row.append(DPP.DecomposedPrimalProblem(self.problem_data,
                                       self.lambda_matrix[i][j], j,
                                       self.y_master, self.DPP_sol[i]))
            self.problem_DPP.append(problem_DPP_row)

###############################################################################
# PRIVATE init_solution()
###############################################################################
    def init_solution(self):
        counter = 0
        s = gurobipy.tupledict()
        tr = gurobipy.tupledict()
        r = gurobipy.tupledict()
        er = gurobipy.tupledict()
        e = gurobipy.tupledict()
        Tlen = self.problem_data.get_names("wildfire")
        Ilen = self.problem_data.get_names("resources")
        first=1
        for res in Ilen:
            for tt in Tlen:
                if (first == 1) :
                    value = 1
                else:
                    value = 0

                s[res,tt] = value
                tr[res,tt] = value
                r[res,tt] = value
                er[res,tt] = value
                e[res,tt] = value
            counter = counter + 1
        variables = gurobipy.tupledict()
        variables["s"] = s
        variables["tr"] = tr
        variables["r"] = r
        variables["er"] = er
        variables["e"] = e
        model = gurobipy.Model("Init")
        solution = _sol.Solution(model, variables)
        #   mu[res,tt] = DPP.get_variables().get_variable('mu')[res,tt]
        return solution

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
