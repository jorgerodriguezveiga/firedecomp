"""Module with guroi information."""

status_info = {
    1: {"status_code": "LOADED", "description": "Model is loaded, but no solution information is available."},
    2: {"status_code": "OPTIMAL", "description": "Model was solved to optimality (subject to tolerances), and an optimal solution is available."},
    3: {"status_code": "INFEASIBLE", "description": "Model was proven to be infeasible."},
    4: {"status_code": "INF_OR_UNBD", "description": "Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize."},
    5: {"status_code": "UNBOUNDED", "description": "Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize."},
    6: {"status_code": "CUTOFF", "description": "Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available."},
    7: {"status_code": "ITERATION_LIMIT", "description": "Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter."},
    8: {"status_code": "NODE_LIMIT", "description": "Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter."},
    9: {"status_code": "TIME_LIMIT", "description": "Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter."},
    10: {"status_code": "SOLUTION_LIMIT", "description": "Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter."},
    11: {"status_code": "INTERRUPTED", "description": "Optimization was terminated by the user."},
    12: {"status_code": "NUMERIC", "description": "Optimization was terminated due to unrecoverable numerical difficulties."},
    13: {"status_code": "SUBOPTIMAL", "description": "Unable to satisfy optimality tolerances; a sub-optimal solution is available."},
    14: {"status_code": "INPROGRESS", "description": "An asynchronous optimization call was made, but the associated optimization run is not yet complete."},
    15: {"status_code": "USER_OBJ_LIMIT", "description": "User specified an objective limit (a bound on either the best objective or the best bound), and that limit has been reached."}
}