"""Module with wildfire suppression model definition."""

# Python packages
import gurobipy
import re
from io import StringIO
import sys

from . import model


# Master ----------------------------------------------------------------------
class Master(model.Model):
    """Master variables will be:
    - y: containment.
    - s: start.
    - e: end.
    - mu: slack if there are not enought resources.
    - tr_start_end: los vuelos asociados al comienzo y fin de actividad.
          ( En el paper definiremos 0 <= tr = tr_start_end + tr_rest <= 1 )
    Auxiliar variables:
    - u: use. (s <--> e)
    - w: work. (u - tr)
    - l: working line build.
    - z: select.
    """
    def __init__(self, problem_data, valid_constraints=None):
        self.model_name = "Master"
        self.obj_zeta = None
        super(Master, self).__init__(
            problem_data=problem_data, relaxed=False,
            valid_constraints=valid_constraints)

    def __build_s__(self):
        data = self.data

        s_ub = {(i, t): 1 if (data.ITW[i] is not True) or (t == data.min_t)
                else 0 for i in data.I for t in data.T}
        # cr definition condition
        s_ub.update({
            (i, t): 0 for i in data.I for t in data.T
            if (data.IOW[i] == 1) and
               (t > data.min_t) and
               (t < data.min_t + data.RP[i])})

        self.variables.s = self.model.addVars(
            data.I, data.T, vtype=gurobipy.GRB.BINARY, lb=0, ub=s_ub,
            name="start")

    def __build_w__(self):
        data = self.data
        u = self.variables.u
        tr = self.variables.tr

        self.variables.w = {
            (i, t): u[i, t] - tr[i, t]
            for i in data.I for t in data.T
        }

    def __build_variables__(self):
        """Build variables."""
        self.__build_s__()
        self.__build_e__()
        self.__build_tr__()

        self.__build_u__()
        self.__build_w__()
        self.__build_z__()

        self.__build_y__()
        self.__build_mu__()

        self.__build_fix_cost_resources__()
        self.__build_variable_cost_resources__()
        self.__build_wildfire_cost__()
        self.__build_law_cost__()

    def __build_logical_3__(self):
        data = self.data
        tr = self.variables.tr
        u = self.variables.u

        self.model.addConstrs(
            (tr[i, t] <= u[i, t]
             for i in data.I for t in data.T),
            name='logical_3')

    def __build_constraints__(self):
        """Build constraints."""
        # Benders constraints
        # -------------------
        self.constraints.opt_int = {}
        self.constraints.opt_start_int = {}
        self.constraints.content_feas_int = {}
        self.constraints.feas_int = {}

        # Valid Constraints
        # -----------------
        self.__build_valid_constraint_contention__()
        self.__build_valid_constraint_work__()

        # Wildfire Containment
        # --------------------
        self.__build_wildfire_containment_1__()
        self.__build_wildfire_containment_2__()

        # Start of Activity
        # -----------------
        self.__build_start_activity_1__()
        self.__build_start_activity_2__()
        self.__build_start_activity_3__()

        # End of Activity
        # ---------------
        self.__build_end_activity__()

        # Breaks
        # ------
        # self.__build_breaks_1_lb__()
        # self.__build_breaks_1_ub__()
        # self.__build_breaks_2__()
        # self.__build_breaks_3__()
        # self.__build_breaks_4__()

        # Maximum Number of Usage Periods in a Day
        # ----------------------------------------
        self.__build_max_usage_periods__()

        # Non-Negligence of Fronts
        # ------------------------
        self.__build_non_negligence_1__()
        self.__build_non_negligence_2__()

        # Logical constraints
        # ------------------------
        self.__build_logical_1__()
        self.__build_logical_2__()
        self.__build_logical_3__()  # Considering only travel periods
        self.__build_logical_4__()

    def update_resources_wildfire(self):
        # Todo mirar que partes hay que actualizar (fijarse en )
        self.problem_data.resources_wildfire.update(
            {(i, t): {
                    'start': round(
                        self.get_var_val("s", i, t)) == 1,
                    'end': round(
                        self.get_var_val("e", i, t)) == 1,
                    'use': round(
                        self.get_var_val("u", i, t)) == 1,
                    'work': round(
                        self.get_var_val("w", i, t)) == 1,
                    'travel': round(
                        self.get_var_val("tr", i, t)) == 1
                }
             for i in self.data.I for t in self.data.T}
        )

    def update_groups_wildfire(self):
        self.problem_data.groups_wildfire.update(
            {
                (g, t): {'num_left_resources': self.get_var_val("mu", g, t)}
                for g in self.data.G for t in self.data.T
            }
        )

    def add_opt_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.opt_int)
        self.constraints.opt_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items())
            + self.variables.zeta >= rhs,
            name='opt_int[{}]'.format(cut_num)
        )
        m.update()

    def getVarByName(self, varname):
        expression = "(?P<name>[^\[\]\s]+)(?:\[(?P<index>.*)\])?"
        search = re.search(expression, varname)
        name = search.group("name")
        index = search.group("index").split(",")
        for k, i in enumerate(index):
            try:
                index[k] = int(i)
            except Exception:
                pass

        index = tuple(index)
        return getattr(self.variables, name)[index]

    def add_opt_start_int_cut(self, i, t, vars_coeffs, rhs):
        m = self.model
        self.constraints.opt_int[i, t] = m.addConstr(
            sum(coeff*self.getVarByName(var)
                for var, coeff in vars_coeffs.items()) >= rhs,
            name=f'opt_start_int[{i}, {t}]'
        )
        m.update()

    def add_feas_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.feas_int)
        self.constraints.feas_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items()) >= rhs,
            name='feas_int[{}]'.format(cut_num)
        )
        m.update()

    def add_contention_feas_int_cut(self, vars_coeffs, rhs):
        m = self.model

        cut_num = len(self.constraints.content_feas_int)
        self.constraints.content_feas_int[cut_num] = m.addConstr(
            sum(coeff*m.getVarByName(var)
                for var, coeff in vars_coeffs.items()) >= rhs,
            name='content_feas_int[{}]'.format(cut_num)
        )
        m.update()

    def add_opt_start_end_int_cut(self, vars_coeffs, rhs):
        m = self.model
        cut_num = len(self.constraints.opt_start_int)
        list_lhs = [
            coeff*m.getVarByName(var) for var, coeff in vars_coeffs.items()
            if m.getVarByName(var) is not None and coeff != 0]

        if len(list_lhs) > 0:
            lhs = sum(list_lhs)
            self.constraints.opt_start_int[cut_num] = \
                m.addConstr(lhs >= rhs, name='start_opt_int[{}]'.format(cut_num))
            m.update()

    def get_S_set(self):
        shared_variables = ["start", "end", "work", "containment"]
        return [
            k for k, v in self.solution.items()
            if re.match("|".join(shared_variables)+"\[.*\]", k)
            if v == 1]
# --------------------------------------------------------------------------- #
