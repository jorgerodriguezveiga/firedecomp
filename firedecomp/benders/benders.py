"""Module with wildfire suppression model definition."""

# Python packages
import logging as log
import re
import time

# Package modules
from firedecomp import benders
from firedecomp import original


# Benders ---------------------------------------------------------------------
class Benders(object):
    def __init__(self, problem_data, min_res_penalty=1000000, gap=0.01):
        if problem_data.period_unit is False:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data

        # Benders info
        self.obj_lb = -float("inf")
        self.obj_ub = float("inf")
        self.gap = gap
        self.iter = 0
        self.time = 0

        self.best_sol = None

        # Models
        self.relaxed_problem = original.model.InputModel(
            problem_data, relaxed=True, min_res_penalty=min_res_penalty)
        self.master = benders.master.Master(
            problem_data, min_res_penalty)
        self.subproblem = benders.subproblem.Subproblem(
            problem_data, min_res_penalty, relaxed=False)
        self.subproblem_relaxed = benders.subproblem.Subproblem(
            problem_data, min_res_penalty, relaxed=True)
        self.subproblem_infeas = benders.subproblem.Subproblem(
            problem_data, min_res_penalty, relaxed=False, slack=True)

        # Master info
        self.L = None

        self.common_vars = list(
            set([v.VarName for v in self.master.model.getVars()]).intersection(
            set([v.VarName for v in self.subproblem.model.getVars()])))

        self.optimality_cuts = []
        self.feasible_cuts = []
        self.integer_cuts = []

    def __log__(self, level="critical"):
        log_level = getattr(log, level)
        if not hasattr(self, "log"):
            self.log = log.getLogger()
            self.log.setLevel(log_level)
            ch = log.StreamHandler()
            formatter = log.Formatter('%(levelname)8s:%(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)
        else:
            for h in self.log.handlers:
                h.setLevel(log_level)

    def get_obj_bound(self):
        model = self.relaxed_problem.solve(None)
        return model.model.getObjective().getValue()

    def add_opt_cut(self):
        matrix = self.subproblem_relaxed.get_constraint_matrix()
        dual = self.subproblem_relaxed.get_dual()
        rhs = self.subproblem_relaxed.get_rhs()
        coeffs = matrix[self.common_vars].T.dot(dual)
        self.master.add_opt_cut(coeffs.to_dict(), rhs.dot(dual))

    def add_opt_int_cut(self):
        lower_obj = self.L
        sub_obj = self.subproblem.get_obj()
        term = lower_obj-sub_obj
        s_set = self.master.get_S_set()
        coeffs = {v: term if v in s_set else -term
                  for v in self.common_vars}
        rhs = term*(len(s_set)-1)+lower_obj
        self.master.add_opt_int_cut(coeffs, rhs)

    def add_contention_feas_int_cut(self):
        S_set = self.master.get_S_set()
        max_t = int(max(self.problem_data.get_names("wildfire")))
        vars = []
        for v in S_set:
            search = re.search('start\[([^,]+),(\d+)\]', v)
            if search:
                res, min_t = search.groups()
                vars += ["start[{},{}]".format(res, t)
                         for t in range(int(min_t), max_t+1)]
            else:
                raise ValueError(
                    "Unkown format of variables start '{}'".format(v))
        coeffs = {v: - 1 if v in vars else 1 for v in self.common_vars}
        rhs = 1 - len(S_set)
        self.master.add_contention_feas_int_cut(coeffs, rhs)

    def add_feas_int_cut(self):
        S_set = self.master.get_S_set()
        coeffs = {v: - 1 if v in S_set else 1 for v in self.common_vars}
        rhs = 1 - len(S_set)
        self.master.add_feas_int_cut(coeffs, rhs)

    def solve(self, max_iters=10,
              solver_options_master=None, solver_options_subproblem=None,
              log_level="CRITICAL"):

        self.__log__(log_level)
        start_time = time.time()
        self.L = self.get_obj_bound()

        master = self.master
        subproblem = self.subproblem
        subproblem_relaxed = self.subproblem_relaxed
        subproblem_infeas = self.subproblem_infeas

        for it in range(1, max_iters+1):
            self.iter = it
            self.log.info("[ITER]: {}".format(it))

            # Solve Master problem
            self.log.debug("\t[MASTER]:")
            master_status = master.solve(solver_options=solver_options_master)

            if master_status != 2:
                self.log.debug("\t - Not optimal solution.")
                return master_status
            else:
                self.log.debug("\t - Optimal")
                start = self.problem_data.resources_wildfire.get_info("start")
                self.log.debug(
                    "\t - Solution: " +
                    ", ".join([str(k) for k, v in start.items() if v == 1]))

            # Update subproblems
            subproblem.__update_model__()
            subproblem_relaxed.__update_model__()
            subproblem_infeas.__update_model__()

            # Solve Relaxed Subproblem
            self.log.debug("\t[RELAXED SUBPROBLEM]:")
            sub_rel_status = subproblem_relaxed.solve(
                solver_options=solver_options_subproblem)
            sub_status = 3
            if sub_rel_status == 2:
                self.log.debug("\t - Optimal")
                self.log.info("\t - Add optimality cut")
                self.add_opt_cut()

                # Solve Subproblem
                self.log.debug("\t[SUBPROBLEM]:")
                sub_status = subproblem.solve(
                    solver_options=solver_options_subproblem)

                if sub_status == 2:
                    self.log.debug("\t - Optimal")

                    self.log.info("\t - Add integer optimality cut")
                    self.add_opt_int_cut()

                    self.log.info("\t - Update objective bounds")
                    self.obj_lb = master.get_obj()
                    new_ub = master.get_obj(zeta=False) + subproblem.get_obj()
                    if new_ub <= self.obj_ub:
                        log.info("\t - Update best solution")
                        self.obj_ub = new_ub
                        self.best_sol = \
                            self.problem_data.resources_wildfire.get_info(
                                "start")

            if sub_status != 2:
                self.log.debug("\t - Not optimal")

                # Solve Subproblem Infeasibilities
                self.log.debug("\t[SLACK SUBPROBLEM]:")
                sub_infeas_status = subproblem_infeas.solve(
                    solver_options=solver_options_subproblem)

                if sub_infeas_status == 2:
                    self.log.debug("\t - Optimal")
                    self.log.info("\t - Add contention integer feasibility cut")
                    self.add_contention_feas_int_cut()
                else:
                    self.log.debug("\t - Not optimal")
                    self.log.info("\t - Add integer feasibility cut")
                    self.add_feas_int_cut()

            self.log.info("[STOP CRITERIA]:")
            self.log.debug("\t - lb: %s", self.obj_lb)
            self.log.debug("\t - ub: %s", self.obj_ub)
            self.log.info("\t - GAP: %s", self.obj_ub - self.obj_lb)

            if (self.obj_ub - self.obj_lb)/(self.obj_ub + self.gap) <= self.gap:
                self.log.info("\t - Convergence")
                break

        self.time = time.time() - start_time
# --------------------------------------------------------------------------- #
