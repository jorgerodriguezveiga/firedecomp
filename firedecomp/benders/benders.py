"""Module with wildfire suppression model definition."""

# Python packages
import logging as log
import re
import time

# Package modules
from firedecomp import benders
from firedecomp import original
from firedecomp import logging
from firedecomp.benders import utils


# Benders ---------------------------------------------------------------------
class Benders(object):
    def __init__(
            self,
            problem_data,
            min_res_penalty=1000000,
            mip_gap_obj=0.01, mip_gap_cost=0.01, max_iters=10, max_time=3600,
            solver_options_master=None,
            solver_options_subproblem=None,
            log_level="benders"
    ):
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data

        # Benders info
        self.obj_lb = -float("inf")
        self.obj_cost_lb = 0
        self.obj_ub = float("inf")
        self.obj_cost_ub = float("inf")
        self.mip_gap_obj = mip_gap_obj
        self.mip_gap_cost = mip_gap_cost
        self.max_iters = max_iters
        self.max_time = max_time
        self.solver_options_master = solver_options_master
        self.solver_options_subproblem = solver_options_subproblem

        self.status = 1
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
        self.subproblem_infeas = benders.subproblem.Subproblem(
            problem_data, min_res_penalty, relaxed=False, slack=True)

        # Master info
        self.L = None

        self.common_vars = [
            v.VarName
            for v in self.master.model.getVars()
            if re.search('start\[([^,]+),(\d+)\]', v.VarName)]

        # Logger
        self.__log__(log_level)

    def __log__(self, level="benders"):
        log.addLevelName(60, "benders")
        log.Logger.benders = logging.benders

        if level != 'benders':
            log_level = getattr(log, level)
            logger = log.getLogger('benders_logging')
            logger.setLevel(log_level)
            logger.addFilter(logging.BendersFilter())
            if len(logger.handlers) == 0:
                ch = log.StreamHandler()
                ch.setLevel(log_level)
                # create formatter and add it to the handlers
                formatter = log.Formatter("%(levelname)8s: %(message)s")
                ch.setFormatter(formatter)
                logger.addHandler(ch)
        else:
            log_level = 60
            logger = log.getLogger('benders')
            logger.setLevel(log_level)
            if len(logger.handlers) == 0:
                ch = log.StreamHandler()
                ch.setLevel(log_level)
                # create formatter and add it to the handlers
                formatter = log.Formatter("%(message)s")
                ch.setFormatter(formatter)
                logger.addHandler(ch)

        self.log = logger

    def get_obj_bound(self):
        model = self.relaxed_problem.solve(None)
        return model.model.getObjective().getValue()

    def add_opt_int_cut(self):
        lower_obj = 0
        sub_obj = self.subproblem.get_obj()
        master_obj = self.master.get_obj(zeta=False)
        zeta_value = sub_obj-master_obj
        term = lower_obj-zeta_value
        s_set = self.master.get_S_set()
        coeffs = {v: term if v in s_set else -term
                  for v in self.common_vars}
        rhs = term*(len(s_set)-1)+lower_obj
        self.master.add_opt_int_cut(coeffs, rhs)

    def add_opt_start_int_cut(self):
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

        sub_obj = self.subproblem.get_obj(law=False)
        master_obj = self.master.get_obj(law=False, zeta=False)
        zeta_value = 0.1*(sub_obj-master_obj)
        term = zeta_value
        s_set = self.master.get_S_set()
        coeffs = {v: term if v in vars else -term
                  for v in self.common_vars}
        rhs = term*(len(s_set)-1)
        self.master.add_opt_start_int_cut(coeffs, rhs)

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

    def solve(self):
        start_time = time.time()
        self.L = self.get_obj_bound()

        master = self.master
        subproblem = self.subproblem
        subproblem_infeas = self.subproblem_infeas

        self.log.benders(utils.format_benders([
            "ITER",
            "SECONDS",
            "LB",
            "UB",
            "MIP_GAP",
            "MIP_GAP_R",
            "OPT_C",
            "FEA_CONT_C",
            "FEA_SING_C"
        ]))

        self.log.benders("-+-".join(["-"*10]*9))
        for it in range(1, self.max_iters+1):
            self.iter = it
            self.log.info("[ITER]: {}".format(self.iter))

            # Solve Master problem
            self.log.debug("\t[MASTER]:")
            master_status = master.solve(
                solver_options=self.solver_options_master)

            if master_status == 3:
                self.log.debug("\t - Not optimal solution.")
                self.status = 3
                break
            else:
                self.log.debug("\t - Optimal")
                start = self.problem_data.resources_wildfire.get_info("start")
                self.log.debug(
                    "\t - Solution: " +
                    ", ".join([str(k) for k, v in start.items() if v == 1]))

            # Update subproblems
            subproblem.__update_model__()

            # Solve Subproblem
            self.log.debug("\t[SUBPROBLEM]:")
            sub_status = subproblem.solve(
                solver_options=self.solver_options_subproblem)

            if sub_status == 2:
                self.log.debug("\t - Optimal")

                self.log.info("\t - Add integer optimality cut")
                self.add_opt_int_cut()
                # self.add_opt_start_int_cut()

                self.log.info("\t - Update objective bounds")
                self.obj_lb = master.get_obj()
                self.obj_cost_lb = master.get_obj()
                new_ub = subproblem.get_obj()
                if new_ub <= self.obj_ub:
                    self.log.info("\t - Update best solution")
                    self.obj_ub = new_ub
                    self.best_sol = \
                        self.problem_data.resources_wildfire.get_info("start")
            else:
                self.log.debug("\t - Not optimal")
                # Update subproblem_infeas
                subproblem_infeas.__update_model__()

                # Solve Subproblem Infeasibilities
                self.log.debug("\t[SLACK SUBPROBLEM]:")
                sub_infeas_status = subproblem_infeas.solve(
                    solver_options=self.solver_options_subproblem)

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

            self.time = time.time() - start_time

            self.log.benders(utils.format_benders([
                self.iter,
                self.time,
                self.obj_lb,
                self.obj_ub,
                (self.obj_ub - self.obj_lb),
                (self.obj_ub - self.obj_lb)/self.obj_ub,
                len(master.constraints.opt_int),
                len(master.constraints.content_feas_int),
                len(master.constraints.feas_int)
            ]))

            if (self.obj_ub - self.obj_lb)/self.obj_ub <= self.mip_gap_obj:
                self.log.info("\t - Convergence.")
                self.log.benders("\nConvergence.")
                self.status = 2
                break
            # elif master
            elif self.max_iters == self.iter:
                self.log.info("\t - Maximum number of iterations exceeded.")
                self.log.benders("\nMaximum number of iterations exceeded.")
                self.status = 7
                break
            elif self.time > self.max_time:
                self.log.info("\t - Maximum time exceeded.")
                self.log.benders("\nMaximum time exceeded.")
                self.status = 9
                break

        self.time = time.time() - start_time

        return self.status
# --------------------------------------------------------------------------- #
