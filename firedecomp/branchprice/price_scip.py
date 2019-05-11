# """Module with wildfire suppression model definition."""
#
# # Python packages
# import pyscipopt as scip
# import logging as log
#
# # Package modules
# from firedecomp.classes import solution
# from firedecomp import config
# from firedecomp.branchprice.benders_scip import Problem_model
# from firedecomp.original import model
#
#
# # Class which can have attributes set.
# class Expando(object):
#     """Todo: create a class for each type of set."""
#     pass
#
#
# class data_pricer(object):
#     """Data_pricer class."""
#
#     def __init__(self, zfun, cc_value, sol = None):
#         """Data_pricer class.
#         Args:
#             zfun (double): objective function value
#             cc_value (dict): complicating constraints values
#             cc_value = {
#                 'wildfire_containment_1': 10,
#                 'wildfire_containment_2': 10,
#                 'non_negligence_1':3,
#                 'non_negligence_2':13
#                 }
#         """
#         self.zfun = [zfun]
#         self.cons_value = [cc_value]
#         self.Nsol = range(len(self.zfun))
#         self.solution = [sol]
#
#     def update(self, zfun, cc_value, sol = None):
#         """Data_pricer class.
#
#         Args:
#             problem
#         """
#         self.zfun.append(zfun)
#         self.cons_value.append(cc_value)
#         self.Nsol = range(len(self.zfun))
#         self.solution.append(sol)
#
#
# # Problem_model ------------------------------------------------------------------
# class MasterPrice_model(object):
#     def __init__(self, problem_data, data_pricer):
#         self.data = problem_data
#         self.data_pricer = data_pricer
#         self.variables = Expando()
#         self.constraints = {}
#
#
#     def __build_variables__(self, tipo="C"):
#         """Build variables."""
#         m = self.model
#         data_pricer = self.data_pricer
#         Nsol = data_pricer.Nsol
#         us = {}
#         for s in Nsol:
#                 us[s] = m.addVar(vtype=tipo, lb=0, name="us[%s]"%(s))
#         self.variables.us = us
#
#     def __build_objectives__(self):
#         """Build objective."""
#         m = self.model
#         data_pricer = self.data_pricer
#         Nsol = data_pricer.Nsol
#         us = self.variables.us
#         self.variables.sum_objs = sum([data_pricer.zfun[s]*us[s] for s in Nsol])
#
#         self.objective = m.setObjective(self.variables.sum_objs)
#
#     def __build_complicated_wildfire_containment_1__(self):
#         m = self.model
#         data_pricer = self.data_pricer
#
#         Nsol = data_pricer.Nsol
#         us = self.variables.us
#         cons = m.addCons(sum([data_pricer.cons_value[s]['wildfire_containment_1']*us[s] for s in Nsol]) <= 0, name='wildfire_containment_1', separate = False, modifiable = True)
#         self.constraints['wildfire_containment_1'] = cons
#
#     def __build_complicated_wildfire_containment_2__(self):
#         m = self.model
#         data = self.data
#         data_pricer = self.data_pricer
#
#         Nsol = data_pricer.Nsol
#         us = self.variables.us
#         cons = {}
#         for t in data.T:
#             cons[t] = m.addCons(sum([data_pricer.cons_value[s]['wildfire_containment_2'][t]*us[s] for s in Nsol]) <= 0, name='wildfire_containment_2[%s]'%t, separate = False, modifiable = True)
#         self.constraints['wildfire_containment_2'] = cons
#
#     def __build_complicated_non_negligence_1__(self):
#         m = self.model
#         data = self.data
#         data_pricer = self.data_pricer
#
#         Nsol = data_pricer.Nsol
#         us = self.variables.us
#         cons = {}
#         for g in data.G:
#             for t in data.T:
#                 cons[g,t] = m.addCons(sum([data_pricer.cons_value[s]['non_negligence_1'][g,t]*us[s] for s in Nsol]) <= 0, name='non_negligence_1[%s,%s]'%(g,t),separate = False, modifiable = True)
#         self.constraints['non_negligence_1'] = cons
#
#     def __build_complicated_non_negligence_2__(self):
#         m = self.model
#         data = self.data
#         data_pricer = self.data_pricer
#
#         Nsol = data_pricer.Nsol
#         us = self.variables.us
#         cons = {}
#         for g in data.G:
#             for t in data.T:
#                 cons[g,t] = m.addCons(sum([data_pricer.cons_value[s]['non_negligence_2'][g,t]*us[s] for s in Nsol]) <= 0, name='non_negligence_2[%s,%s]'%(g,t),separate = False, modifiable = True)
#         self.constraints['non_negligence_2'] = cons
#
#     def __build_convexity__(self):
#         m = self.model
#         data_pricer = self.data_pricer
#
#         Nsol = data_pricer.Nsol
#         us = self.variables.us
#         cons = m.addCons(sum([us[s] for s in Nsol]) == 1, name='convexity',separate = False, modifiable = True)
#         self.constraints['convexity'] = cons
#
#     def __ensure_feasibility__(self):
#         m = self.model
#         ub_fixed = 10000
#         bigM = 100000
#
#         self.variables.w = m.addVar(vtype = "C", lb = 0, ub = ub_fixed, obj = bigM, name="w")
#         v = {}
#         for i in self.constraints:
#             cons = self.constraints[i]
#             if isinstance(cons,dict):
#                 vdict = {}
#                 for j in cons:
#                     vdict[j] = m.addVar(vtype = "C", lb = 0, ub = ub_fixed, obj = bigM, name='v[%s,%s]'%(str(i),str(j)))
#                     m.addConsCoeff(cons[j], vdict[j], 1)
#                     m.addConsCoeff(cons[j], self.variables.w, -1)
#                 v[i] = vdict
#             else:
#                 v[i] = m.addVar(vtype = "C", lb = 0, ub = ub_fixed, obj = bigM, name='v[%s]'%str(i))
#                 m.addConsCoeff(cons, v[i], 1)
#                 m.addConsCoeff(cons, self.variables.w, -1)
#
#     def __build_constraints__(self):
#         self.__build_complicated_wildfire_containment_1__()
#         self.__build_complicated_wildfire_containment_2__()
#         self.__build_complicated_non_negligence_1__()
#         self.__build_complicated_non_negligence_2__()
#
#     def build_model(self):
#         self.model = scip.Model("MasterPricer")
#         self.__build_variables__()
#         self.__build_objectives__()
#         self.__build_constraints__()
#         self.__ensure_feasibility__()
#         self.__build_convexity__()
#
#     def build_binary_model(self):
#         self.model = scip.Model("MasterPricer")
#         self.__build_variables__(tipo="B")
#         self.__build_objectives__()
#         self.__build_constraints__()
#         self.__ensure_feasibility__()
#         self.__build_convexity__()
#
#
# def computeCCvalue(data,model,variables):
#
#
#     # Solution values
#     y  = {t: model.getVal(variables.y[t]) for t in data.T+[data.min_t-1]}
#     mu = {(g, t): model.getVal(variables.mu[g, t]) for g in data.G for t in data.T}
#
#     uval = {(i,t): sum([model.getVal(variables.s[i, tind]) for tind in data.T_int(p_max=t)])
#                  - sum([model.getVal(variables.e[i, tind]) for tind in data.T_int(p_max=t-1)])
#                 for i in data.I for t in data.T}
#     w = {(i, t): uval[i, t] -
#                  model.getVal(variables.r[i, t])-
#                  model.getVal(variables.tr[i, t]) for i in data.I for t in data.T}
#     z = {i: sum([model.getVal(variables.e[i, t]) for t in data.T]) for i in data.I}
#
#     #########################################
#     # Function value of the original problem
#     #########################################
#     obj_orig = 0
#     obj_orig += sum([data.C[i]*uval[i, t] for i in data.I for t in data.T])
#     obj_orig += sum([data.P[i] * z[i] for i in data.I])
#     obj_orig += sum([data.NVC[t] * y[t-1] for t in data.T])
#     obj_orig += sum([data.Mp*mu[g, t] for g in data.G for t in data.T])
#
#     ###########################
#     # Complicating constraints
#     ###########################
#     cons_value = {}
#     # wildfire_containment_1
#     cons_value['wildfire_containment_1'] = (sum([data.PER[t]*y[t-1] for t in data.T])
#         - sum([data.PR[i, t]*w[i, t] for i in data.I for t in data.T]))
#
#     # wildfire_containment_2
#     val = {}
#     for t in data.T:
#         val[t] = (sum([data.PER[t1] for t1 in data.T_int(p_max=t)])*y[t-1]
#                - sum([data.PR[i, t1]*w[i, t1] for i in data.I for t1 in data.T_int(p_max=t)])
#                - data.M*y[t])
#     cons_value['wildfire_containment_2'] = val
#
#      # 'non_negligence_1'
#     val = {}
#     for g in data.G:
#         for t in data.T:
#             val[g,t] = data.nMin[g, t]*y[t-1] - mu[g, t] - sum([w[i, t] for i in data.Ig[g]])
#     cons_value['non_negligence_1'] = val
#
#     # 'non_negligence_2'
#     val = {}
#     for g in data.G:
#         for t in data.T:
#             val[g,t] = sum([w[i, t] for i in data.Ig[g]]) - data.nMax[g, t]*y[t-1]
#     cons_value['non_negligence_2'] = val
#
#     return obj_orig, cons_value
#
#
# def computeOBJpricer(data,model,variables):
#
#     u = {(i,t): sum([model.getVal(variables.s[i, tind]) for tind in data.T_int(p_max=t)])
#                  - sum([model.getVal(variables.e[i, tind]) for tind in data.T_int(p_max=t-1)])
#                 for i in data.I for t in data.T}
#     y  = {t: model.getVal(variables.y[t]) for t in data.T+[data.min_t-1]}
#     mu = {(g, t): model.getVal(variables.mu[g, t]) for g in data.G for t in data.T}
#     z = {i: sum([model.getVal(variables.e[i, t]) for t in data.T]) for i in data.I}
#
#     obj_total = 0
#     obj_total += sum([data.C[i]*u[i, t] for i in data.I for t in data.T])
#     obj_total += sum([data.P[i] * z[i] for i in data.I])
#     obj_total += sum([data.NVC[t] * y[t-1] for t in data.T])
#     obj_total += sum([data.Mp*mu[g, t] for g in data.G for t in data.T])
#
#     return obj_total
#
# def convexSolution(solutionlist,usol):
#
#     # Initialize finalsol to a dictionary of variables with 0 values
#     finalsol = {}
#     sol = solutionlist[-1]
#     for variable_name in sol:
#         var = sol[variable_name]
#         zero_var = {}
#         for key in var:
#             zero_var[key] = 0
#         finalsol[variable_name] = zero_var
#
#
#     # Weighted sum of the variables:
#     for counter,val in enumerate(usol):
#         if val > 0:
#             sol = solutionlist[counter]
#             for variable_name in sol:
#                 var_dict = sol[variable_name]
#                 finalvar_dict = finalsol[variable_name]
#                 for key in finalvar_dict:
#                     finalvar_dict[key] += val*var_dict[key]
#
#     # If they are close to be integer put it integer
#     for variable_name in finalsol:
#         finalvar_dict = finalsol[variable_name]
#         for key in finalvar_dict:
#             val = finalvar_dict[key]
#             if(abs(val-int(val)<1e-8)):
#                 finalvar_dict[key] = int(val)
#
#     return finalsol
#
#
# class CutPricer(scip.Pricer):
#
#     # The reduced cost function for the variable pricer
#     def pricerredcost(self):
#
#         # Problem Data
#         #print('*********Llego PRICER')
#         data = self.data['problem_data']
#         data_pricer = self.data['data_pricer']
#         data_cons   = self.data['constraints']
#         masterModel = self.model
#
#         #######################################################
#         # Retrieving the dual solutions of the Master Solution
#         dualSolutions = {}
#
#         # wildfire_containment_1
#         dualSolutions['wildfire_containment_1'] = masterModel.getDualMultiplier(data_cons['wildfire_containment_1'])
#
#         # 'wildfire_containment_2'
#         duals = {}
#         for t in data.T:
#             cons = data_cons['wildfire_containment_2'][t]
#             duals[t] = masterModel.getDualMultiplier(cons)
#         dualSolutions['wildfire_containment_2'] = duals
#
#         # non_negligence_1
#         duals = {}
#         for g in data.G:
#             for t in data.T:
#                 cons = data_cons['non_negligence_1'][g,t]
#                 duals[g,t] = masterModel.getDualMultiplier(cons)
#         dualSolutions['non_negligence_1'] = duals
#
#         # non_negligence_2
#         duals = {}
#         for g in data.G:
#             for t in data.T:
#                 cons = data_cons['non_negligence_2'][g,t]
#                 duals[g,t] = masterModel.getDualMultiplier(cons)
#         dualSolutions['non_negligence_2'] = duals
#
#         # convexity:
#         dualSolutions['convexity'] = masterModel.getDualMultiplier(data_cons['convexity'])
#
#         # Building a MIP to solve the subproblem
#         subMIP_problem = Problem_model(data,dual_value = dualSolutions)
#         subMIP = subMIP_problem.get_subproblem_pricer_model()
#
#         # Turning off presolve
#         subMIP.setPresolve(scip.SCIP_PARAMSETTING.OFF)
#
#         # Setting the verbosity level to 0
#         subMIP.hideOutput()
#
#         # Solving the subMIP
#         subMIP.optimize()
# #        subMIP.writeProblem('subproblema.lp')
# #        subMIP.writeBestSol('solprueba.sol',write_zeros=True)
# #        masterModel.writeProblem('masterModel.lp')
# #        masterModel.writeBestSol('master.sol',write_zeros=True)
#
#         objsubMIP = subMIP.getObjVal()
#         #objval = computeOBJpricer(data,subMIP,subMIP_problem.variables)
#         print('objval ',objsubMIP, dualSolutions['convexity'])
#         #import pdb; pdb.set_trace()
#
#         # Adding the column to the master problem
#         if objsubMIP + 1e-4 < dualSolutions['convexity']:
#             #print('Entro')
#             currentNumVar = len(data_pricer.cons_value)
#
#             # Update data_pricer object:
#             zfun, cc_value = computeCCvalue(data,subMIP,subMIP_problem.variables)
#             sol = subMIP_problem.get_solution_variables_all()
#             data_pricer.update(zfun, cc_value, sol)
#             #subMIP.writeProblem('subproblema[%s].lp'%str(currentNumVar))
#             #subMIP.writeBestSol('solMIP[%s].sol'%str(currentNumVar))
#             print('zfun ',zfun)
#             print('cc_value ',cc_value)
#
#             # Creating new var; must set pricedVar to True
#             us_new = self.model.addVar("us[%s]"%(str(currentNumVar)), vtype = "C", obj = zfun, pricedVar = True)
#             self.data['variables'].append(us_new)
#
#             # Adding the new variable to the constraints of the master problem
#             # wildfire_containment_1
#             cons = data_cons['wildfire_containment_1']
#             coeff = cc_value['wildfire_containment_1']
#             self.model.addConsCoeff(cons, us_new, coeff)
#
#             # wildfire_containment_2
#             for t in data.T:
#                 cons = data_cons['wildfire_containment_2'][t]
#                 coeff = cc_value['wildfire_containment_2'][t]
#                 self.model.addConsCoeff(cons, us_new, coeff)
#
#             # non_negligence_1
#             for g in data.G:
#                 for t in data.T:
#                     cons = data_cons['non_negligence_1'][g,t]
#                     coeff = cc_value['non_negligence_1'][g,t]
#                     self.model.addConsCoeff(cons, us_new, coeff)
#
#             # non_negligence_2
#             for g in data.G:
#                 for t in data.T:
#                     cons = data_cons['non_negligence_2'][g,t]
#                     coeff = cc_value['non_negligence_2'][g,t]
#                     self.model.addConsCoeff(cons, us_new, coeff)
#
#             # convexity
#             cons = data_cons['convexity']
#             coeff = 1
#             self.model.addConsCoeff(cons, us_new, coeff)
#
#
#         return {'result':scip.SCIP_RESULT.SUCCESS}
#
#     # The initialisation function for the variable pricer to retrieve the transformed constraints of the problem
#     def pricerinit(self):
#         for i in self.data['constraints']:
#             cons = self.data['constraints'][i]
#             if isinstance(cons,dict):
#                 for j in cons:
#                     self.data['constraints'][i][j] = self.model.getTransformedCons(cons[j])
#             else:
#                 self.data['constraints'][i] = self.model.getTransformedCons(cons)
#
#
# def solve_BranchPrice(problem_data):
#
#     data = problem_data.data
# #
# #    #######################
# #    ## Sol Gurobi
# #    #######################
# #    gurobi_problem = model.InputModel(problem_data)
# #    m = gurobi_problem.solve(None)
# #    sol_gurobi = {'s':{(i,t): m.variables.s[i, t].x for i in data.I for t in data.T},
# #                  'tr':{(i,t): m.variables.tr[i, t].x for i in data.I for t in data.T},
# #                  'r':{(i,t): m.variables.r[i, t].x for i in data.I for t in data.T},
# #                  'er':{(i,t): m.variables.er[i, t].x for i in data.I for t in data.T},
# #                  'e':{(i,t): m.variables.e[i, t].x for i in data.I for t in data.T},
# #                  'u':{(i,t): m.variables.u[i, t].getValue() for i in data.I for t in data.T},
# #                  'w':{(i,t): m.variables.w[i, t].getValue() for i in data.I for t in data.T},
# #                  'z':{(i): m.variables.z[i].getValue() for i in data.I}
# #                 }
# #
# #    usol = {(i, t): sum([sol_gurobi['s'][i,tind] for tind in data.T_int(p_max=t)])
# #                  - sum([sol_gurobi['e'][i,tind] for tind in data.T_int(p_max=t-1)])
# #            for i in data.I for t in data.T}
# #    wsol = {(i, t): usol[i, t] - sol_gurobi['r'][i, t] - sol_gurobi['tr'][i, t] for i in data.I for t in data.T}
# #    for t in  problem_data.data.T:
# #        for i in data.I:
# #            print(sol_gurobi['w'][i,t],' === ', wsol[i,t])
# #
# #    sol = sol_gurobi
# #    for t in  problem_data.data.T:
# #        print('************************************Periodo = ',t)
# #        print('**********************    s     tr      r    er      e      u    w      z')
# #        for i in  problem_data.data.I:
# #            print('AVION = ',i, '=> ',sol['s'][i,t], ' ', sol['tr'][i,t],' ',sol['r'][i,t], ' ',sol['er'][i,t], '  ',sol['e'][i,t], '  ',sol['u'][i,t], '  ',sol['w'][i,t], '  ',sol['z'][i])
# #
# #
# #    import pdb; pdb.set_trace()
#     #######################
#     ## Sol SCIP
#     #######################
#     original_problem = Problem_model(problem_data.data)
#     original = original_problem.get_original_model()
#     original.optimize()
#     sol_orig = original_problem.get_solution_variables_all()
#
#
#     sol = sol_orig
#     for t in  problem_data.data.T:
#         print('************************************Periodo = ',t, 'Contention = ',sol['y'][t])
#         print('**********************    s     tr      r    er      e      u    w      z')
#         for i in  problem_data.data.I:
#             print('AVION = ',i, '=> ',sol['s'][i,t], ' ', sol['tr'][i,t],' ',sol['r'][i,t], ' ',sol['er'][i,t], '  ',sol['e'][i,t], '  ',sol['u'][i,t], '  ',sol['w'][i,t], '  ',sol['z'][i])
#
#
#     import pdb; pdb.set_trace()
#
#     #######################
#     ## Sol with PRICER
#     #######################
#
#     zfun = original.getObjVal()
#     obj, cc_value = computeCCvalue(data,original,original_problem.variables)
#
#     #data_price = data_pricer(zfun,cc_value)
#     #data_price.update(zfun,cc_value)
#
#
# #    # Initialize Data Pricer
#     zfun = 1000000
#     dualval = 10
#
#     cc_value = {}
#      # wildfire_containment_1
#     cc_value['wildfire_containment_1'] = dualval
#
#     # wildfire_containment_2
#     val = {}
#     for t in data.T:
#         val[t] = dualval
#     cc_value['wildfire_containment_2'] = val
#
#      # 'non_negligence_1'
#     val = {}
#     for g in data.G:
#         for t in data.T:
#             val[g,t] = dualval
#     cc_value['non_negligence_1'] = val
#
#     # 'non_negligence_2'
#     val = {}
#     for g in data.G:
#         for t in data.T:
#             val[g,t] = dualval
#     cc_value['non_negligence_2'] = val
#     data_price = data_pricer(zfun,cc_value)
#
#     ####################################################3
#     zfun = 1000000
#     dualval = 5
#
#     cc_value = {}
#      # wildfire_containment_1
#     cc_value['wildfire_containment_1'] = dualval
#
#     # wildfire_containment_2
#     val = {}
#     for t in data.T:
#         val[t] = dualval
#     cc_value['wildfire_containment_2'] = val
#
#      # 'non_negligence_1'
#     val = {}
#     for g in data.G:
#         for t in data.T:
#             val[g,t] = dualval
#     cc_value['non_negligence_1'] = val
#
#     # 'non_negligence_2'
#     val = {}
#     for g in data.G:
#         for t in data.T:
#             val[g,t] = dualval
#     cc_value['non_negligence_2'] = val
#     data_price.update(zfun,cc_value)
#
#     #☺ Creaste Master Model:
#     masterModel = MasterPrice_model(problem_data.data,data_price)
#     masterModel.build_model()
#     s = masterModel.model
#
#     # Remove Presolve
#     s.setPresolve(0)
#
#     # creating a pricer
#     pricer = CutPricer()
#     s.includePricer(pricer, "LumesPricer", "Divide the problem by resource")
#
#     # Setting the pricer_data for use in the init and redcost functions
#     pricer.data = {}
#     pricer.data['problem_data'] = problem_data.data
#     pricer.data['data_pricer']  = masterModel.data_pricer
#     pricer.data['constraints']  = masterModel.constraints
#     pricer.data['variables']    = [masterModel.variables.us[s] for s in masterModel.data_pricer.Nsol]
#
#     # solve problem
#     s.optimize()
#     s.writeBestSol('master.sol',write_zeros=True)
#
#     solutionlist =  pricer.data['data_pricer'].solution
#     usol = [s.getVal(var) for var in pricer.data['variables']]
#     finalsol = convexSolution(solutionlist,usol)
#
#
#     masterBinModel = MasterPrice_model(problem_data.data,pricer.data['data_pricer'])
#     masterBinModel.build_binary_model()
#     sBin = masterBinModel.model
#     sBin.optimize()
#     sBin.writeProblem('masterBin.lp')
#     sBin.writeBestSol('masterBin.sol',write_zeros=True)
#
#
#     sol = finalsol
#     for t in  problem_data.data.T:
#         print('************************************Periodo = ',t, 'Contention = ',sol['y'][t])
#         print('**********************    s     tr      r    er      e      u    w      z')
#         for i in  problem_data.data.I:
#             print('AVION = ',i, '=> ',sol['s'][i,t], ' ', sol['tr'][i,t],' ',sol['r'][i,t], ' ',sol['er'][i,t], '  ',sol['e'][i,t], '  ',sol['u'][i,t], '  ',sol['w'][i,t], '  ',sol['z'][i])
#
#     update_problem_with_sol(problem_data,finalsol)
#
#
#
# def update_problem_with_sol(problem_data,sol):
#
#     # Load variables values
#     data = problem_data.data
#     problem_data.resources.update(
#         {i: {'select': sum([sol['e'][i,t] for t in data.T])== 1}
#          for i in data.I})
#
#     uval = {(i,t): sum([sol['s'][i, tind] for tind in data.T_int(p_max=t)])
#                  - sum([sol['e'][i, tind] for tind in data.T_int(p_max=t-1)])
#             for i in data.I for t in data.T}
#     zval = {(i,t): uval[i,t] - sol['r'][i, t] - sol['tr'][i, t]
#             for i in data.I for t in data.T}
#
#     problem_data.resources_wildfire.update(
#         {(i, t): {
#             'start': sol['s'][i, t] == 1,
#             'travel': sol['tr'][i, t] == 1,
#             'rest': sol['r'][i, t] == 1,
#             'end_rest': sol['er'][i, t] == 1,
#             'end': sol['e'][i, t] == 1,
#             'use':  uval[i,t] == 1,
#             'work': zval[i, t] == 1
#         }
#          for i in data.I for t in data.T})
#
#     problem_data.groups_wildfire.update(
#         {(g, t): {'num_left_resources': sol['mu'][g, t]}
#          for g in data.G for t in data.T})
#
#     contained = {t: sol['y'][t] == 0 for t in data.T}
#     contained_period = [t for t, v in contained.items()
#                         if v is True]
#
#     if len(contained_period) > 0:
#         first_contained = min(contained_period) + 1
#     else:
#         first_contained = data.max_t + 1
#
#     problem_data.wildfire.update(
#         {t: {'contained': False if t < first_contained else True}
#          for t in data.T})