"""Module with utilities."""

import pandas as pd


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


# get_rhs --------------------------------------------------------------
def get_rhs(m):
    rhs_dict = {}
    for constr in m.getConstrs():
        con_name = constr.ConstrName
        rhs_dict[con_name] = constr.RHS
    return rhs_dict
# --------------------------------------------------------------------------- #
