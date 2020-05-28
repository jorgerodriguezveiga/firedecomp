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
        self.obj_lb_prev = -float("inf")
        self.obj_cost_lb = 0
        self.obj_ub = float("inf")
        self.obj_cost_ub = float("inf")
        self.mip_gap_obj = mip_gap_obj
        self.mip_gap_cost = mip_gap_cost
        self.max_iters = max_iters
        self.max_time = max_time
        self.solver_options_master = solver_options_master
        self.solver_options_subproblem = solver_options_subproblem
        self.constrvio = float('inf')

        self.prev_num_cuts = 0
        self.num_cuts = 0

        self.status = 1
        self.iter = 0
        self.time = 0

        self.best_sol = None

        # Models
        self.relaxed_problem = original.model.InputModel(
            problem_data, relaxed=True)
        self.master = benders.master.Master(
            problem_data)
        self.subproblem = benders.subproblem.Subproblem(
            problem_data, slack=True)
        # self.subproblem_infeas = benders.subproblem.Subproblem(
        #     problem_data, relaxed=False, slack=True)

        # Master info
        self.L = None

        self.common_vars = [
            v.VarName
            for v in self.master.model.getVars()
            if re.search('start\[([^,]+),(\d+)\]', v.VarName)]

        self.__start_info__ = {}

        # Logger
        self.log_level = log_level
        if self.log_level == 'benders':
            self.tbl = lambda *args: print(utils.format_benders(*args))

    def get_obj_bound(self):
        model = self.relaxed_problem.solve(None)
        return model.model.getObjective().getValue()

    def add_opt_int_cut(self):
        """Corte optimalidad variables enteras."""
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

    def add_contention_feas_int_cut(self):
        """Factibilidad enteras.

        ToDo: mirar que esten bien.
        """
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

    def add_opt_start_int_cut(self, i, t):
        work = self.__start_info__[i, t]['work']
        max_t = int(max(self.problem_data.get_names("wildfire")))
        coeffs = {'s[{},{}]'.format(i, t): - max_t}
        coeffs.update({"w[{},{}]".format(k[0], k[1]): - 1
                       for k, v in work.items() if v == 0})
        rhs = - max_t
        self.master.add_opt_start_int_cut(i, t, coeffs, rhs)

    def get_start_resource_info(self):
        """Add start info of the selected resource and period.
        """
        for r in self.problem_data.resources:
            if r.select:
                rp = r.__resource_period__
                start = rp.get_info("start")
                index = max(start, key=start.get)
                if index not in self.__start_info__:
                    self.__start_info__[index] = {
                        'work': rp.get_info("work"),
                        'travel': rp.get_info("travel"),
                        'rest': rp.get_info("rest"),
                        'end_rest': rp.get_info("end_rest"),
                    }
                    self.add_opt_start_int_cut(*index)


    def update_activity(self):
        start = {
            rp.resource.get_index(): rp.period.get_index()
            for rp in self.problem_data.resources_wildfire
            if rp.start
        }
        self.problem_data.resources_wildfire.update(
            {(i, t):
             {
                'rest': self.__start_info__[i, start[i]]['rest'][i, t],
                'end_rest': self.__start_info__[i, start[i]]['end_rest'][i, t],
                'travel': self.__start_info__[i, start[i]]['travel'][i, t],
                'work': self.__start_info__[i, start[i]]['work'][i, t]
             }
             if self.problem_data.resources_wildfire[i, t].use else
             {
                'rest': False,
                'end_rest': False,
                'travel': False,
                'work': False
             }
             for i, t in self.problem_data.resources_wildfire.get_names()
            }
        )

    def solve(self):
        start_time = time.time()
        self.L = self.get_obj_bound()

        master = self.master
        subproblem = self.subproblem
        # subproblem_infeas = self.subproblem_infeas

        self.tbl("-+-".join(["----------"]*6))
        self.tbl(
            "ITER",
            "SECONDS",
            "LB",
            "UB",
            "OPTI_CUT",
            "FEAS_CUT"
        )
        self.tbl("-+-".join(["----------"]*6))

        for it in range(1, self.max_iters+1):
            self.iter = it
            log.info("[ITER]: {}".format(self.iter))

            # Solve Master problem
            log.debug("\t[MASTER]:")
            master_status = master.solve(
                solver_options=self.solver_options_master)
            try:
                self.constrvio = self.master.get_constrvio()
            except:
                self.constrvio = float('inf')
            if master_status == 3:
                log.debug("\t - Not optimal solution.")
                self.tbl(
                    self.iter,
                    self.time,
                    self.obj_lb,
                    self.obj_ub,
                    len(master.constraints.opt_int) +
                    len(master.constraints.opt_start_int),
                    len(master.constraints.content_feas_int) +
                    len(master.constraints.feas_int)
                )
                self.status = 3
                break
            else:
                log.debug("\t - Optimal")
                start = self.problem_data.resources_wildfire.get_info("start")
                log.debug(
                    "\t - Solution: " +
                    ", ".join([str(k) for k, v in start.items() if v == 1]))

            # Solve Subproblem
            log.debug("\t[SUBPROBLEM]:")
            for r in self.problem_data.resources:
                if r.select:
                    rp = r.__resource_period__
                    start = rp.get_info("start")
                    index = max(start, key=start.get)
                    if index not in self.__start_info__:
                        resources = [index[0]]
                        # Update subproblems
                        subproblem.__update_model__(resources=resources)

                        # Solve
                        sub_status = subproblem.solve(
                            resources=resources,
                            solver_options=self.solver_options_subproblem)

                        if sub_status == 2:
                            self.__start_info__[index] = {
                                'work': rp.get_info("work"),
                                'travel': rp.get_info("travel"),
                                'rest': rp.get_info("rest"),
                                'end_rest': rp.get_info("end_rest"),
                            }
                            log.info("\t - Add integer optimality cut")
                            self.add_opt_start_int_cut(*index)
                        else:
                            log.debug("\t - Not optimal solution.")

            self.obj_lb_prev = self.obj_lb
            self.obj_lb = master.get_obj()
            log.info("[STOP CRITERIA]:")
            log.debug(f"\t - curr_obj: {self.obj_lb}")
            log.debug(f"\t - prev_obj: {self.obj_lb_prev}")
            log.info(f"\t - GAP: {self.obj_lb - self.obj_lb_prev}")

            self.time = time.time() - start_time

            self.tbl(
                self.iter,
                self.time,
                self.obj_lb,
                self.obj_lb_prev,
                len(master.constraints.opt_int) +
                len(master.constraints.opt_start_int),
                len(master.constraints.content_feas_int) +
                len(master.constraints.feas_int)
            )

            self.prev_num_cuts = self.num_cuts
            self.num_cuts = len(master.constraints.opt_int) + \
                len(master.constraints.opt_start_int) + \
                len(master.constraints.content_feas_int) + \
                len(master.constraints.feas_int)

            if self.prev_num_cuts == self.num_cuts:
                log.info("\t - Convergence.")
                log.info("\t - Update solution")
                self.update_activity()
                master.solve(solver_options=self.solver_options_master)
                self.status = 2
                break
            # elif master
            elif self.max_iters == self.iter:
                log.info("\t - Maximum number of iterations exceeded.")
                log.info("\t - Update solution")
                self.update_activity()
                master.solve(solver_options=self.solver_options_master)
                self.status = 7
                break
            elif self.time > self.max_time:
                log.info("\t - Maximum time exceeded.")
                log.info("\t - Update solution")
                self.update_activity()
                master.solve(solver_options=self.solver_options_master)
                self.status = 9
                break

        self.time = time.time() - start_time

        return self.status
# --------------------------------------------------------------------------- #
