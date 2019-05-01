"""Module with utilities."""

import numpy as np
import math


# get_expr_cos ----------------------------------------------------------------
def get_expr_cos(expr):
    for i in range(expr.size()):
        dvar = expr.getVar(i)
        yield expr.getCoeff(i), dvar
# --------------------------------------------------------------------------- #


# get_matrix_coo --------------------------------------------------------------
def get_matrix_coo(m):
    coeff_dict = {}
    for constr in m.getConstrs():
        con_name = constr.ConstrName
        coeff_dict[con_name] = {}
        for coeff, var in get_expr_cos(m.getRow(constr)):
            var_name = var.VarName
            coeff_dict[con_name][var_name] = coeff
    return coeff_dict
# --------------------------------------------------------------------------- #


# get_rhs ---------------------------------------------------------------------
def get_rhs(m):
    rhs_dict = {}
    for constr in m.getConstrs():
        con_name = constr.ConstrName
        rhs_dict[con_name] = constr.RHS
    return rhs_dict
# --------------------------------------------------------------------------- #


# get_start_info --------------------------------------------------------------
def get_start_info(data, start_dict):
    """Establish work, travel and rest periods."""
    work = {(i, t): 0 for i in data.I for t in data.T}
    rest = {(i, t): 0 for i in data.I for t in data.T}
    travel = {(i, t): 0 for i in data.I for t in data.T}

    for i in data.I:
        if (data.ITW[i] is not True) and (data.IOW[i] is not True):
            work_count = 1
        else:
            if start_dict[i, data.min_t] == 1:
                work_count = data.CWP[i] - data.CRP[i] + 1
            else:
                work_count = data.WP[i] + 1

        start = False
        travel_count = 1
        for t in data.T:
            if start_dict[i, t] == 1:
                start = True

            if start:
                if t == data.max_t:
                    work_count += 1
                    travel_count += 1
                    travel[i, t] = 1
                    continue

                if (data.ITW[i] is True) or (data.IOW[i] is True):
                    if (work_count == data.CWP[i] + 1) and (data.A[i] == 1):
                        work_count += 1
                        rest[i, t] = 1
                        continue

                if work_count <= data.WP[i] - data.TRP[i]:
                    work_count += 1
                    if travel_count <= data.A[i]:
                        travel_count += 1
                        travel[i, t] = 1
                    else:
                        work[i, t] = 1
                elif work_count <= data.WP[i]:
                    work_count += 1
                    travel_count += 1
                    travel[i, t] = 1
                elif work_count <= data.WP[i] + data.RP[i]:
                    work_count += 1
                    rest[i, t] = 1
                elif work_count < data.WP[i] + data.RP[i]:
                    work_count += 1
                    travel_count += 1
                    travel[i, t] = 1
                else:
                    work_count = 0
                    travel_count += 1
                    travel[i, t] = 1
            else:
                pass

    return {'work': work, 'rest': rest, 'travel': travel}
# --------------------------------------------------------------------------- #


# get_start_info --------------------------------------------------------------
def get_start_resource_info(data, i, start_period):
    """Establish work, travel and rest periods."""
    work = {(i, t): 0 for t in data.T}
    rest = {(i, t): 0 for t in data.T}
    travel = {(i, t): 0 for t in data.T}

    if (data.ITW[i] == True) or (data.IOW[i] == True):
        if start_period == data.min_t:
            work_count = data.CWP[i] - data.CRP[i] + 1
        else:
            work_count = data.WP[i] + 1
    else:
        work_count = 1

    start = False
    travel_count = 1
    for t in data.T:
        if start_period == t:
            start = True

        if start:
            if t >= data.max_t - data.TRP[i] + 1:
                work_count += 1
                travel_count += 1
                travel[i, t] = 1
                continue

            if (data.ITW[i] is True) or (data.IOW[i] is True):
                if (work_count == data.CWP[i] + 1) and (data.A[i] == 1):
                    work_count += 1
                    rest[i, t] = 1
                    continue

            if work_count <= data.WP[i] - data.TRP[i]:
                work_count += 1
                if travel_count <= data.A[i]:
                    travel_count += 1
                    travel[i, t] = 1
                else:
                    work[i, t] = 1
            elif work_count <= data.WP[i]:
                work_count += 1
                travel[i, t] = 1
            elif work_count <= data.WP[i] + data.RP[i]:
                work_count += 1
                rest[i, t] = 1
            elif work_count < data.WP[i] + data.RP[i] + data.TRP[i]:
                work_count += 1
                travel[i, t] = 1
            else:
                work_count = 1 + data.TRP[i]
                travel[i, t] = 1
        else:
            pass

    return {'work': work, 'rest': rest, 'travel': travel}
# --------------------------------------------------------------------------- #


# get_minimum_resources -------------------------------------------------------
def get_minimum_resources(data):
    """Get a lower bound of the minimum number of resources."""
    resources_performance = {
        i: sum([data.PR[i, t]*data.start_work[i, t]
                for t in data.T]) for i in data.I}
    max_wildfire_perimeter = max(data.PER)

    num = len([
        i
        for i, v in enumerate(np.cumsum(sorted(
            resources_performance.values(), reverse=True)))
        if v <= max_wildfire_perimeter])

    return num
# --------------------------------------------------------------------------- #


# format_fix_work --------------------------------------------------------------
def format_fix_work(message):
    return " | ".join([format_fix_work_element(e) for e in message])
# --------------------------------------------------------------------------- #


# format_fix_work_element ------------------------------------------------------
def format_fix_work_element(element):
    log_s_format = "{:10s}"
    log_e_format = "{:.4E}"
    log_f_format = "{:7.2f}"
    log_d_format = "{:10d}"
    if isinstance(element, int):
        return log_d_format.format(element)
    elif isinstance(element, float):
        if abs(element) == float("inf") or math.isnan(element):
            str_num = str(element)
            return " " * (10 - len(str_num)-3) + str_num + " "*3
        else:
            if len(str(round(element))) > 7:
                return log_e_format.format(element)
            else:
                str_num = log_f_format.format(element)
                return " "*(10-len(str_num)) + str_num
    else:
        return log_s_format.format(str(element))
# --------------------------------------------------------------------------- #
