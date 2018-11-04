"""Module with wildfire suppression model definition."""

# Python packages
import logging as log

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
            problem_data, min_res_penalty, relaxed=True, slack=True)

        # Master info
        self.L = None

        self.common_vars = list(
            set([v.VarName for v in self.master.model.getVars()]).intersection(
            set([v.VarName for v in self.subproblem.model.getVars()])))

        self.optimality_cuts = []
        self.feasible_cuts = []
        self.integer_cuts = []

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

    def add_feas_cut(self):
        matrix = self.subproblem_infeas.get_constraint_matrix()
        dual = self.subproblem_infeas.get_dual()

        rhs = self.subproblem_infeas.get_rhs()
        coeffs = matrix[self.common_vars].T.dot(dual)
        self.master.add_feas_cut(coeffs.to_dict(), rhs.dot(dual))

    def add_feas_int_cut(self):
        S_set = self.master.get_S_set()
        coeffs = {v: - 1 if v in S_set else 1 for v in self.common_vars}
        rhs = 1 - len(S_set)
        self.master.add_feas_int_cut(coeffs, rhs)

    def solve(self, max_iters=10,
              solver_options_master=None, solver_options_subproblem=None):

        self.L = self.get_obj_bound()

        master = self.master
        subproblem = self.subproblem
        subproblem_relaxed = self.subproblem_relaxed
        subproblem_infeas = self.subproblem_infeas

        for iter in range(1, max_iters+1):
            log.info("Iteration: {}".format(iter))

            # Solve Master problem
            log.debug("\t[MASTER] Solve")
            master_status = master.solve(solver_options=solver_options_master)
            log.debug(
                {k: v
                 for k, v in self.problem_data.resources_wildfire.get_info(
                    "start").items() if v == 1})
            if master_status != 2:
                log.debug("\t[MASTER] Not optimal solution.")
                return master_status
            else:
                log.debug("\t[MASTER] Optimal")

            # Update subproblems
            subproblem.__update_model__()
            subproblem_relaxed.__update_model__()
            subproblem_infeas.__update_model__()

            # Solve Relaxed Subproblem
            log.debug("\t[SUBREL] Solve")
            sub_rel_status = subproblem_relaxed.solve(
                solver_options=solver_options_subproblem)
            if sub_rel_status == 2:
                log.debug("\t[SUBREL] Optimal")
                log.info("\t[MASTER] Add optimality cut")
                self.add_opt_cut()

                # Solve Subproblem
                log.debug("\t[SUBPRO] Solve")
                sub_status = subproblem.solve(
                    solver_options=solver_options_subproblem)

                if sub_status == 2:
                    log.debug("\t[SUBPRO] Optimal")

                    log.info("\t[BENDER] Update obj bounds")
                    self.obj_lb = master.get_obj()
                    self.obj_ub = min(self.obj_ub, subproblem.get_obj())

                    log.info("\t[MASTER] Add integer optimality cut")
                    self.add_opt_int_cut()
                else:
                    log.debug("\t[SUBPRO] Not optimal")
                    log.info("\t[MASTER] Add integer feasibility cut")
                    self.add_feas_int_cut()

            else:
                log.debug("\t[SUBREL] Not optimal")

                # Solve Subproblem Infeasibilities
                log.debug("\t[SUBINF] Solve")
                sub_infeas_status = subproblem_infeas.solve(
                    solver_options=solver_options_subproblem)

                if sub_infeas_status == 2:
                    log.info("\t[MASTER] Add feasibility cut")
                    # Todo: Check theses constraints
                    # self.add_feas_cut()

                    log.info("\t[MASTER] Add integer feasibility cut")
                    self.add_feas_int_cut()
                else:
                    raise ValueError("[SUBINF] Must be optimal")

            if (self.obj_ub - self.obj_lb)/(self.obj_ub + self.gap) <= self.gap:
                log.info("[BENDER] Convergence")
                break

            log.info("[BENDER] lb = {}, ub = {}".format(
                self.obj_lb, self.obj_ub))
# --------------------------------------------------------------------------- #
