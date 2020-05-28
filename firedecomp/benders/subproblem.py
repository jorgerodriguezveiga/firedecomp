"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import pandas as pd
from io import StringIO
import sys
import copy
import logging as log

# Package modules
from firedecomp.benders import utils
from . import model


# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Subproblem ------------------------------------------------------------------
class Subproblem(model.Model):
    """Subproblem:

    Variables:
        - r: rest.
        - er: end rest.
        - tr: travel.
        - w = u - r - tr

    Todo: Considered only selected resources.
    """

    def __init__(self, problem_data, slack=False):
        self.problem_data = problem_data
        self.data = self.problem_data.data
        self.subproblem_resources = {
            i: SubproblemResource(problem_data, i, slack=slack)
            for i in self.data.I
        }

    def solve(self, resources, solver_options):
        """Solve mathematical model.

        Args:
            resources (:obj:`list`): list of resources.
            solver_options (:obj:`dict`): gurobi options. Default ``None``.
                Example: ``{'TimeLimit': 10}``.
        """
        status = []
        for i in resources:
            pr = self.subproblem_resources[i]
            log.info(f"Solve subproblem for resource: {i}")
            status.append(pr.solve(solver_options=solver_options))
        return 2 if all([s == 2 for s in status]) else status

    def __update_model__(self, resources):
        for i in resources:
            pr = self.subproblem_resources[i]
            log.info(f"Update subproblem for resource: {i}")
            pr.__update_model__()

# --------------------------------------------------------------------------- #


# SubproblemResource ----------------------------------------------------------
class SubproblemResource(model.Model):
    """Subproblem:

    Variables:
        - r: rest.
        - er: end rest.
        - tr: travel.
        - w = u - r - tr

    Todo: Considered only selected resources.
    """

    def __init__(self, problem_data, resource, slack=False):
        self.model_name = "Subproblem"
        self.slack = slack
        super(SubproblemResource, self).__init__(
            problem_data=problem_data, relaxed=False, valid_constraints=[])
        self.data = copy.copy(self.data)
        self.data.I = [resource]

    def __build_variables__(self):
        """Build variables."""
        if self.slack:
            self.__build_slack__()

        # Fix
        self.__build_s__()
        self.__build_e__()

        # Variables
        self.__build_r__()
        self.__build_tr__()
        self.__build_er__()

        # Compute
        self.__build_u__()
        self.__build_w__()
        self.__build_cr__()

    def __build_slack__(self):
        data = self.data

        self.variables.slack_pos = self.model.addVars(
            data.I, data.T, name="slack_pos", lb=0, ub=1)
        self.variables.slack_neg = self.model.addVars(
            data.I, data.T, name="slack_neg", lb=0, ub=1)

    def __build_s__(self):
        problem = self.problem_data
        slack_pos = self.variables.slack_pos
        slack_neg = self.variables.slack_neg

        self.variables.s = {
            i: float(v == 1) + slack_pos[i] - slack_neg[i]
            if self.slack else
            float(v == 1)
            for i, v in problem.resources_wildfire.get_info("start").items()
            if i[0] in self.data.I
        }

    def __build_e__(self):
        data = self.data
        problem = self.problem_data
        s_cum = []
        e_ub = {
            i: sum(s_cum)
            for i, v in problem.resources_wildfire.get_info("start").items()
            if (s_cum.append(v is not None and v == 1) or True)
            and (i[0] in self.data.I)
        }
        var_info = self.binary_variables_type()

        self.variables.e = self.model.addVars(
            data.I, data.T, name="end",
            lb=var_info['lb'], ub=e_ub, vtype=var_info['vtype'])

    def __update_model__(self):
        """Update model. The best way found is building the model again."""
        self.__build_model__()

    def __build_objective__(self):
        """Build objective."""
        m = self.model
        data = self.data
        w = self.variables.w
        if self.slack:
            m.setObjective(
                sum(w[i, t] for i in data.I for t in data.T)
                - 1000 * sum(self.variables.slack_pos[i, t]
                             for i in data.I for t in data.T)
                - 1000 * sum(self.variables.slack_neg[i, t]
                             for i in data.I for t in data.T),
                gurobipy.GRB.MAXIMIZE
            )
        else:
            m.setObjective(
                sum(w[i, t] for i in data.I for t in data.T),
                gurobipy.GRB.MAXIMIZE
            )

    def __build_constraints__(self):
        """Build constraints."""
        # Valid Constraints
        # -----------------
        # self.__build_valid_constraint_contention__()
        # self.__build_valid_constraint_work__()

        # Wildfire Containment
        # --------------------
        # self.__build_wildfire_containment_1__()
        # self.__build_wildfire_containment_2__()

        # Start of Activity
        # -----------------
        self.__build_start_activity_1__()
        # self.__build_start_activity_2__()
        # self.__build_start_activity_3__()

        # End of Activity
        # ---------------
        self.__build_end_activity__()

        # Breaks
        # ------
        self.__build_breaks_1_lb__()
        self.__build_breaks_1_ub__()
        self.__build_breaks_2__()
        self.__build_breaks_3__()
        self.__build_breaks_4__()

        # Maximum Number of Usage Periods in a Day
        # ----------------------------------------
        # self.__build_max_usage_periods__()

        # Non-Negligence of Fronts
        # ------------------------
        # self.__build_non_negligence_1__()
        # self.__build_non_negligence_2__()

        # Logical constraints
        # ------------------------
        # self.__build_logical_1__()
        # self.__build_logical_2__()
        self.__build_logical_3__()
        # self.__build_logical_4__()

    def get_constraint_matrix(self):
        """Get constraint matrix."""
        return pd.DataFrame(utils.get_matrix_coo(self.model)).fillna(0).T

    def get_rhs(self):
        return pd.Series(utils.get_rhs(self.model))

    def update_problem_with_solution(self):
        self.solution = {v.VarName: v.x for v in self.model.getVars()}

        # Load variables values
        if not self.relaxed:
            self.update_resources_wildfire()

    def update_resources_wildfire(self):
        infeas = [
            i
            for i in self.data.I
            if sum([
                self.get_var_val("slack_pos", i, t) +
                self.get_var_val("slack_neg", i, t)
                for t in self.data.T
            ]) > 0
        ]
        self.problem_data.resources_wildfire.update(
            {
                (i, t): {
                    'travel': round(
                        self.get_var_val("tr", i, t)) == 1,
                    'rest': round(
                        self.get_var_val("r", i, t)) == 1,
                    'end_rest': round(
                        self.get_var_val("er", i, t)) == 1,
                    'work': round(
                        self.get_var_val("w", i, t)) == 1
                }
                if i not in infeas else
                {
                    'travel': False,
                    'rest': False,
                    'end_rest': False,
                    'work': False
                }
                for i in self.data.I for t in self.data.T
        })
# --------------------------------------------------------------------------- #
