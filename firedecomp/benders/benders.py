"""Module with wildfire suppression model definition."""

# Python packages
import logging as log
import re
import time
import numpy as np

# Package modules
from firedecomp import benders
from firedecomp import logging
from firedecomp.benders import utils


# Benders ---------------------------------------------------------------------
class Benders(object):
    def __init__(
            self,
            problem_data,
            mip_gap_obj=0.01, max_iters=10, max_time=3600,
            n_start_info=0,
            start_period=None,
            step_period=6,
            solver_options_master=None,
            log_level="benders"
    ):
        """Initialize the Benders object.

        Args:
            problem_data (:obj:`Problem`): problem data.
            mip_gap_obj (:obj:`float`): MIP GAP tolerance for stop criteria.
                Defaults to 0.01.
            max_iters (:obj:`int`): maximum number of iterations. Defaults to
                10.
            max_time (:obj:`float`): maximum cpu time (in seconds). Defaults to
                3600.
            n_start_info (:obj:`int`): number periods to compute start
                information of the resources. A large number implies a less
                number of iterations but the execution time cost increase due
                to the high number of cuts. Defaults to 0.
            start_period (:obj:`int`): number of periods to start the algorithm.
                Iterations over the number of periods are done until to reach
                the global optimal solution. If ``None`` the maximum period is
                taken. Defaults to ``None``.
            step_period (:obj:`int`): step between the start_period and the
                next period to solve the period_problem. Defaults to ``5``.
            solver_options_master (:obj:`dict`): dictionary with solver options
                for the master problem. If ``None`` no options. Defaults to
                ``None``.
            log_level (:obj:`str`): logging level. Defaults to ``'benders'``.
        """
        if problem_data.period_unit is not True:
            raise ValueError("Time unit of the problem is not a period.")

        self.problem_data = problem_data

        # Benders info
        self.obj_lb = -float("inf")
        self.obj_ub = float("inf")
        self.mip_gap_obj = mip_gap_obj
        self.max_iters = max_iters
        self.max_time = max_time
        self.n_start_info = n_start_info
        self.start_period = start_period
        self.step_period = step_period
        self.solver_options_master = solver_options_master

        self.num_cuts_prev = 0
        self.best_obj_period = float("inf")
        self.obj = float("inf")
        self.status = 1
        self.period_status = 1
        self.iter = 0
        self.time = 0
        self.runtime = 0
        self.__start_time__ = time.time()

        self.best_sol = None

        # Models
        self.problem_data.update_period_data(max_period=self.start_period)
        self.master = benders.master.Master(
            self.problem_data)
        self.subproblem = benders.subproblem.Subproblem(
            self.problem_data, relaxed=False)
        self.subproblem_infeas = benders.subproblem.Subproblem(
            self.problem_data, relaxed=False, slack=True)

        # Master info
        self.common_vars = [
            v.VarName
            for v in self.master.model.getVars()
            if re.search('start\[([^,]+),(\d+)\]', v.VarName)]
        self.__start_info__ = {}
        self.master.__start_info__ = self.__start_info__

        # Logger
        self.__log__(log_level)

    def get_start_resource_info(self, i, t):
        """Get start info of the selected resource and period.

        Args:
            i (:obj:`str`): resource name.
            t (:obj:`int`): period name.
        """
        if (i, t) not in self.__start_info__:
            self.__start_info__[i, t] = utils.get_start_resource_info(
                self.problem_data.data, i, t)
            self.add_opt_start_int_cut(i, t, self.__start_info__[i, t]['work'])
        return self.__start_info__[i, t]

    def get_start_info(self, list_resorce_period, info=None):
        """Get start info of the selected resources and periods.

        Args:
            list_resorce_period (:obj:`list`): list of pairs, where first
                element is the resource name and the second the period name.
            info (:obj:`list`): list of desired information. If ``None`` return
                all information. Allowed options are: ``'work'``, ``'travel`''
                and ``'rest'``. Defaults to ``None``.
        """
        if info is None:
            info = ['work', 'travel', 'rest']

        return {m: {k: v for i, t in list_resorce_period
                    for k, v in self.get_start_resource_info(i, t)[m].items()}
                for m in info}

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

    def add_opt_start_int_cut(self, i, t, work):
        max_t = int(max(self.problem_data.get_names("wildfire")))
        coeffs = {'start[{},{}]'.format(i, t): - max_t}
        coeffs.update({"work[{},{}]".format(k[0], k[1]): - 1
                       for k, v in work.items() if v == 0})
        rhs = - max_t
        self.master.add_opt_start_int_cut(coeffs, rhs)

    def add_contention_feas_int_cut(self):
        S_set = self.master.get_S_set()
        max_t = self.problem_data.data.max_t
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
        """Solve the problem."""
        self.status = 1
        self.runtime = 0
        self.__start_time__ = time.time()
        data = self.problem_data.data
        if 'TimeLimit' not in self.solver_options_master:
            self.solver_options_master['TimeLimit'] = max(1, self.max_time)

        header = utils.format_benders([
            "PER",
            "ITER",
            "SECONDS",
            "OBJ",
            "LB_PER",
            "START_C",
            "FEA_CONT_C",
            "FEA_SING_C"
        ])
        sep = "-+-".join(["-"*10]*8)

        self.log.benders(header)
        self.log.benders(sep)

        new_obj = float('inf')
        periods = [
            p
            for p in np.arange(self.start_period, data.max_t, self.step_period)
        ] + [data.max_t]

        warm_start = False

        for period in periods:
            self.log.info("Period: {}".format(period))
            self.best_obj_period = new_obj
            if self.best_obj_period < float('inf'):
                self.master.max_obj = self.best_obj_period

            self.solver_options_master['TimeLimit'] = \
                max(min(self.solver_options_master['TimeLimit'],
                        self.max_time - (time.time() - self.__start_time__)),
                    0)
            status = self.solve_periods(max_period=int(period),
                                        warm_start=warm_start)
            warm_start = True
            if status == 2:
                new_obj = self.problem_data.get_cost()
                if abs(self.best_obj_period - new_obj) <= self.mip_gap_obj:
                    self.log.benders("\nConvergence.")
                    self.status = 2
                    break
                else:
                    if period == data.max_t:
                        self.status = 2
                    else:
                        self.log.benders(sep)
                        self.status = 1
            else:
                self.log.benders("")
                self.status = status

        return self.status

    def solve_periods(self, max_period=None, warm_start=False):

        self.period_status = 1
        if max_period is None:
            max_period = self.start_period

        self.problem_data.update_period_data(max_period=max_period)
        self.master.update_model()

        master = self.master
        if warm_start:
            self.master.use_warm_start()

        min_t = self.problem_data.data.min_t

        self.get_start_info(
            list_resorce_period=[
                (i, t)
                for t in range(min_t, min_t+self.n_start_info)
                for i in self.problem_data.data.I
            ]
        )

        self.num_cuts_prev = \
            len(master.constraints.opt_start_int) + \
            len(master.constraints.content_feas_int) + \
            len(master.constraints.feas_int)

        for it in range(1, self.max_iters+1):
            self.iter = it
            self.log.info("[ITER]: {}".format(self.iter))

            # Solve Master problem
            self.log.debug("\t[MASTER]:")
            master_status = master.solve(
                solver_options=self.solver_options_master)
            self.runtime += master.model.runtime

            if master_status == 3:
                self.log.debug("\t - Not optimal solution.")
                self.period_status = 3
                break
            else:
                self.log.debug("\t - Optimal")
                start = self.problem_data.resources_wildfire.get_info("start")
                self.log.debug(
                    "\t - Solution: " +
                    ", ".join([str(k) for k, v in start.items() if v == 1]))

            # Get start info and add cuts if they are needed
            self.get_start_info(
                list_resorce_period=[k for k, v in start.items() if v is True]
            )

            num_cuts = \
                len(master.constraints.opt_start_int) + \
                len(master.constraints.content_feas_int) + \
                len(master.constraints.feas_int)

            if (self.obj_ub - self.obj_lb)/self.obj_ub <= self.mip_gap_obj:
                self.log.info("\t - Convergence.")
                self.obj = self.obj_lb
                self.period_status = 2
            elif self.max_iters == self.iter:
                self.log.info("\t - Maximum number of iterations exceeded.")
                self.period_status = 7
            elif self.time > self.max_time:
                self.log.info("\t - Maximum time exceeded.")
                self.period_status = 9
            elif self.num_cuts_prev == num_cuts:
                self.log.info("\t - Convergence. No cuts added.")
                self.obj = self.obj_lb
                self.period_status = 2

            self.time = time.time() - self.__start_time__

            self.log.info("[STOP CRITERIA]:")
            self.log.debug("\t - lb: %s", self.obj_lb)

            self.log.benders(utils.format_benders([
                max_period,
                self.iter,
                self.time,
                self.obj,
                self.obj_lb,
                len(master.constraints.opt_start_int),
                len(master.constraints.content_feas_int),
                len(master.constraints.feas_int)
            ]))

            if self.period_status != 1:
                break
            self.num_cuts_prev = num_cuts

        self.time = time.time() - self.__start_time__
        return self.period_status
# --------------------------------------------------------------------------- #
