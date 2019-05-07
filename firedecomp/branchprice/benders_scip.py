"""Module with wildfire suppression model definition."""

# Python packages
import pyscipopt as scip
import logging as log
import re as re
import subprocess

# Package modules
from firedecomp.classes import solution
from firedecomp import config

GCG_DIR = "/opt/scipoptsuite-6.0.1/gcg/bin"

# Class which can have attributes set.
class Expando(object):
    """Todo: create a class for each type of set."""
    pass


# Problem_model ------------------------------------------------------------------
class Problem_model(object):
    def __init__(self, problem_data, dual_value = None):

        self.data = problem_data
        self.dual_value = dual_value
        self.variables = Expando()
        self.constraints = Expando()
        
    def __build_variables_all__(self):
        """Build variables."""
        m = self.model
        data = self.data
         # Variables
        # =========
        # Resources
        # ---------
        s = {}
        tr = {}
        r = {}
        er = {}
        e = {}
        for i in data.I:
            for t in data.T:
                s[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="start[%s,%s]"%(i,t))
                tr[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="travel[%s,%s]"%(i,t))
                r[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="rest[%s,%s]"%(i,t))
                er[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="end_rest[%s,%s]"%(i,t))
                e[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="end[%s,%s]"%(i,t))


        # Auxiliar variables
        u = {
            (i, t):
                scip.quicksum(s[i,tind] for tind in data.T_int(p_max=t))
                - scip.quicksum(e[i,tind] for tind in data.T_int(p_max=t-1))
            for i in data.I for t in data.T}
        w = {(i, t): u[i, t] - r[i, t] - tr[i, t] for i in data.I for t in data.T}
        z = {i: scip.quicksum(e[i,tind] for tind in data.T) for i in data.I}

        cr = {(i, t):
              sum([
                  (t+1-t1)*s[i, t1]
                  - (t-t1)*e[i, t1]
                  - r[i, t1]
                  - data.WP[i]*er[i, t1]
                  for t1 in data.T_int(p_max=t)])
              for i in data.I for t in data.T
              if not data.ITW[i] and not data.IOW[i]}

        cr.update({
            (i, t):
                (t+data.CWP[i]-data.CRP[i]) * s[i, data.min_t]
                + sum([
                    (t + 1 - t1 + data.WP[i]) * s[i, t1]
                    for t1 in data.T_int(p_min=data.min_t+1, p_max=t)])
                - sum([
                    (t - t1) * e[i, t1]
                    + r[i, t1]
                    + data.WP[i] * er[i, t1]
                    for t1 in data.T_int(p_max=t)])
            for i in data.I for t in data.T
            if data.ITW[i] or data.IOW[i]})

        # Wildfire
        # --------
        y = {t: 
                m.addVar(vtype="B", lb=0, ub=1,name="contention[%s]"%t) 
                for t in data.T + [data.min_t-1]}
        # start_no_contained
        m.chgVarLb(y[data.min_t-1],1)
        
        mu = {(g,t): 
                m.addVar(vtype="C", lb=0,name="missing_resources[%s,%s]"%(g,t)) 
                for g in data.G for t in data.T}


        self.variables.s = s
        self.variables.tr = tr
        self.variables.r = r
        self.variables.er = er
        self.variables.e = e
        self.variables.u = u
        self.variables.w = w
        self.variables.z = z
        self.variables.cr = cr
        self.variables.y = y
        self.variables.mu = mu

    def __build_variables_master__(self):
        """Build variables."""
        m = self.model
        data = self.data
        s = {}
        e = {}
        for i in data.I:
            for t in data.T:
                s[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="start[%s,%s]"%(i,t))
                e[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="end[%s,%s]"%(i,t))

        # Auxiliar variables
        u = {
            (i, t):
                scip.quicksum(s[i,tind] for tind in data.T_int(p_max=t))
                - scip.quicksum(e[i,tind] for tind in data.T_int(p_max=t-1))
            for i in data.I for t in data.T}
        z = {i: scip.quicksum(e[i,tind] for tind in data.T) for i in data.I}

        self.variables.s = s
        self.variables.e = e
        self.variables.u = u
        self.variables.z = z
    
    def __build_variables_subproblem__(self):
        """Build variables."""
        m = self.model
        data = self.data
        # Variables
        # =========
        # Resources
        # ---------
        s = {}
        tr = {}
        r = {}
        er = {}
        e = {}
        for i in data.I:
            for t in data.T:
                s[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="start[%s,%s]"%(i,t))
                tr[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="travel[%s,%s]"%(i,t))
                r[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="rest[%s,%s]"%(i,t))
                er[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="end_rest[%s,%s]"%(i,t))
                e[i,t] = m.addVar(vtype="B", lb=0, ub=1, name="end[%s,%s]"%(i,t))


        # Auxiliar variables
        u = {
            (i, t):
                scip.quicksum(s[i,tind] for tind in data.T_int(p_max=t))
                - scip.quicksum(e[i,tind] for tind in data.T_int(p_max=t-1))
            for i in data.I for t in data.T}
        w = {(i, t): u[i, t] - r[i, t] - tr[i, t] for i in data.I for t in data.T}
        z = {i: scip.quicksum(e[i,tind] for tind in data.T) for i in data.I}

        cr = {(i, t):
              sum([
                  (t+1-t1)*s[i, t1]
                  - (t-t1)*e[i, t1]
                  - r[i, t1]
                  - data.WP[i]*er[i, t1]
                  for t1 in data.T_int(p_max=t)])
              for i in data.I for t in data.T
              if not data.ITW[i] and not data.IOW[i]}

        cr.update({
            (i, t):
                (t+data.CWP[i]-data.CRP[i]) * s[i, data.min_t]
                + sum([
                    (t + 1 - t1 + data.WP[i]) * s[i, t1]
                    for t1 in data.T_int(p_min=data.min_t+1, p_max=t)])
                - sum([
                    (t - t1) * e[i, t1]
                    + r[i, t1]
                    + data.WP[i] * er[i, t1]
                    for t1 in data.T_int(p_max=t)])
            for i in data.I for t in data.T
            if data.ITW[i] or data.IOW[i]})

        # Wildfire
        # --------
        y = {t: 
                m.addVar(vtype="B", lb=0, ub=1,name="contention[%s]"%t) 
                for t in data.T + [data.min_t-1]}
        # start_no_contained
        m.chgVarLb(y[data.min_t-1],1)
            
        mu = {(g,t): 
                m.addVar(vtype="C", lb=0,name="missing_resources[%s,%s]"%(g,t)) 
                for g in data.G for t in data.T}


        self.variables.s = s
        self.variables.tr = tr
        self.variables.r = r
        self.variables.er = er
        self.variables.e = e
        self.variables.u = u
        self.variables.w = w
        self.variables.z = z
        self.variables.cr = cr
        self.variables.y = y
        self.variables.mu = mu

        
    def __build_objective_all__(self):
        """Build objective."""
        m = self.model
        data = self.data
        z = self.variables.z
        u = self.variables.u
        y = self.variables.y
        mu = self.variables.mu
        
        self.variables.variable_cost_resources = sum([data.C[i]*u[i, t] for i in data.I for t in data.T])
        self.variables.fix_cost_resources = sum([data.P[i] * z[i] for i in data.I])
        self.variables.wildfire_cost = sum([data.NVC[t] * y[t-1] for t in data.T])
        self.variables.law_cost = sum([data.Mp*mu[g, t] for g in data.G for t in data.T])
        
        self.objective = m.setObjective(
            self.variables.variable_cost_resources +
            self.variables.fix_cost_resources +
            self.variables.wildfire_cost +
            self.variables.law_cost)
            
    def __build_objective_master__(self):
        """Build objective."""
        m = self.model
        data = self.data
        z = self.variables.z
        u = self.variables.u
        
        self.variables.variable_cost_resources = sum([data.C[i]*u[i, t] for i in data.I for t in data.T])
        self.variables.fix_cost_resources = sum([data.P[i] * z[i] for i in data.I])
        
        self.objective = m.setObjective(
            self.variables.variable_cost_resources +
            self.variables.fix_cost_resources)
            
    def __build_objective_subproblem__(self):
        """Build objective."""
        m = self.model
        data = self.data
        y = self.variables.y
        mu = self.variables.mu
        
        self.variables.wildfire_cost = sum([data.NVC[t] * y[t-1] for t in data.T])
        self.variables.law_cost = sum([data.Mp*mu[g, t] for g in data.G for t in data.T])
        
        m.setObjective(
            self.variables.wildfire_cost + 
            self.variables.law_cost)
        
    def __build_objective_subproblem_pricer__(self):
        m = self.model
        data = self.data
        z = self.variables.z
        u = self.variables.u
        y = self.variables.y
        w = self.variables.w
        mu = self.variables.mu
        
        self.variables.variable_cost_resources = sum([data.C[i]*u[i, t] for i in data.I for t in data.T])
        self.variables.fix_cost_resources = sum([data.P[i] * z[i] for i in data.I])
        self.variables.wildfire_cost = sum([data.NVC[t] * y[t-1] for t in data.T])
        self.variables.law_cost = sum([data.Mp*mu[g, t] for g in data.G for t in data.T])
        #self.variables.law_cost = sum([100*mu[g, t] for g in data.G for t in data.T])
        
        self.variables.wildfire_containment_1 = -1*self.dual_value['wildfire_containment_1']*(
            sum([data.PER[t]*y[t-1] for t in data.T]) -
            sum([data.PR[i, t]*w[i, t] for i in data.I for t in data.T])
            )
            
        self.variables.wildfire_containment_2 = 0
        for t in data.T:
            self.variables.wildfire_containment_2 += -1*self.dual_value['wildfire_containment_2'][t]*(
                 sum([data.PER[t1] for t1 in data.T_int(p_max=t)])*y[t-1] -
                 sum([data.PR[i, t1]*w[i, t1] for i in data.I for t1 in data.T_int(p_max=t)]) - 
                 data.M*y[t]
                 )
            
        self.variables.non_negligence_1 = 0
        for g in data.G:
            for t in data.T:
                self.variables.non_negligence_1 += -1*self.dual_value['non_negligence_1'][g,t]*(
                    data.nMin[g, t]*y[t-1] - mu[g, t] - sum([w[i, t] for i in data.Ig[g]])     
                    )
                
        self.variables.non_negligence_2 = 0
        for g in data.G:
            for t in data.T:
                self.variables.non_negligence_2 += -1*self.dual_value['non_negligence_2'][g,t]*(
                        sum([w[i, t] for i in data.Ig[g]]) - data.nMax[g, t]*y[t-1]
                        )
                
        self.objective = m.setObjective(
            self.variables.variable_cost_resources +
            self.variables.fix_cost_resources +
            self.variables.wildfire_cost +
            self.variables.law_cost +
            self.variables.wildfire_containment_1 +
            self.variables.wildfire_containment_2 +
            self.variables.non_negligence_1 +
            self.variables.non_negligence_2
            )
    
#    def __build_wildfire_containment_0__(self):
#        m = self.model
#        data = self.data
#        y = self.variables.y
#        m.addCons(y[data.min_t-1] == 1, name='start_no_contained')
        
    def __build_wildfire_containment_1__(self):
        m = self.model
        data = self.data
        y = self.variables.y
        w = self.variables.w

        m.addCons(sum([data.PER[t]*y[t-1] for t in data.T]) <=
                sum([data.PR[i, t]*w[i, t] for i in data.I for t in data.T]),
                name='wildfire_containment_1')
                
    def __build_wildfire_containment_2__(self):
        m = self.model
        data = self.data
        y = self.variables.y
        w = self.variables.w
        
        for t in data.T:
            m.addCons(
                data.M*y[t] >=
                 sum([data.PER[t1] for t1 in data.T_int(p_max=t)])*y[t-1] -
                 sum([data.PR[i, t1]*w[i, t1]
                      for i in data.I for t1 in data.T_int(p_max=t)])
                ,name='wildfire_containment_2[%s]'%t)
       
    def __build_start_activity_1__(self):
        m = self.model
        data = self.data
        w  = self.variables.w
        tr = self.variables.tr
        
        for i in data.I:
            for t in data.T:
                m.addCons(
                    data.A[i]*w[i, t] <=
                     sum([tr[i, t1] for t1 in data.T_int(p_max=t)])
                    ,name='start_activity_1[%s,%s]'%(i,t))
    
    def __build_start_activity_2__(self):
        m = self.model
        data = self.data
        s  = self.variables.s
        z = self.variables.z
        for i in data.I:
            if data.ITW[i] == True:
                m.addCons(
                     s[i, data.min_t] +
                     sum([(data.max_t + 1)*s[i, t]
                          for t in data.T_int(p_min=data.min_t+1)]) <=
                     data.max_t*z[i]
                    ,name='start_activity_2[%s]'%i)
    
    def __build_start_activity_3__(self):
        m = self.model
        data = self.data
        s  = self.variables.s
        z = self.variables.z
        for i in data.I:
            if data.ITW[i] == False:
                m.addCons(
                    sum([s[i, t] for t in data.T]) <= z[i]
                    ,name='start_activity_3[%s]'%i)
        
    def __build_end_activity__(self):
        m = self.model
        data = self.data
        tr  = self.variables.tr
        e = self.variables.e
        
        for i in data.I:
            for t in data.T:
                m.addCons(
                    sum([tr[i, t1] for t1 in data.T_int(p_min=t-data.TRP[i]+1,
                                                                   p_max=t)
                          ]) >= data.TRP[i]*e[i, t]
                    ,name='end_activity[%s,%s]'%(i,t))
        
    def __build_breaks_1__(self):  
        m = self.model
        data = self.data
        cr  = self.variables.cr
        
        for i in data.I:
            for t in data.T:
                m.addCons(
                    0 <= cr[i, t]
                    ,name='Breaks_1_lb[%s,%s]'%(i,t))
                
                m.addCons(
                    cr[i, t] <= data.WP[i]
                    ,name='Breaks_1_ub[%s,%s]'%(i,t))
    
    def __build_breaks_2__(self):  
        m = self.model
        data = self.data
        r  = self.variables.r
        er  = self.variables.er
        for i in data.I:
            for t in data.T:
                m.addCons(
                    r[i, t] <= sum([er[i, t1]
                                     for t1 in data.T_int(p_min=t,
                                                                    p_max=t+data.RP[i]-1)])
                    ,name='Breaks_2[%s,%s]'%(i,t))
                
    def __build_breaks_3__(self): 
        m = self.model
        data = self.data
        r  = self.variables.r
        s  = self.variables.s
        er  = self.variables.er
        
        for i in data.I:
            for t in data.T:
                m.addCons(
                    sum([
                        r[i, t1]
                        for t1 in data.T_int(p_min=t-data.RP[i]+1, p_max=t)]) >=
                     data.RP[i]*er[i, t]
                     if t >= data.min_t - 1 + data.RP[i] else
                     data.CRP[i]*s[i, data.min_t] +
                     sum([r[i, t1] for t1 in data.T_int(p_max=t)]) >=
                     data.RP[i]*er[i, t]
                    ,name='Breaks_3[%s,%s]'%(i,t))
    
    def __build_breaks_4__(self): 
        m = self.model
        data = self.data
        r  = self.variables.r
        tr  = self.variables.tr
        for i in data.I:
            for t in data.T:
                m.addCons(
                    sum([r[i, t1]+tr[i, t1]
                          for t1 in data.T_int(p_min=t-data.TRP[i],
                                                         p_max=t+data.TRP[i])]) >=
                     len(data.T_int(p_min=t-data.TRP[i],
                                              p_max=t+data.TRP[i]))*r[i, t]
                    ,name='Breaks_4[%s,%s]'%(i,t))
                    
    def __build_max_usage_periods__(self):
        m = self.model
        data = self.data
        u  = self.variables.u
        
        for i in data.I:
            m.addCons(
                sum([u[i, t] for t in data.T]) <= data.UP[i] - data.CUP[i]
                ,name='max_usage_periods[%s]'%i)

    def __build_non_negligence_1__(self):
        m = self.model
        data = self.data
        w  = self.variables.w
        y  = self.variables.y
        mu  = self.variables.mu
        
        for g in data.G:
            for t in data.T:
                m.addCons(
                    sum([w[i, t] for i in data.Ig[g]]) >= data.nMin[g, t]*y[t-1] - mu[g, t]
                    ,name='non_negligence_1[%s,%s]'%(g,t))
    
    def __build_non_negligence_2__(self):
        m = self.model
        data = self.data
        w  = self.variables.w
        y  = self.variables.y
        for g in data.G:
            for t in data.T:
                m.addCons(
                    sum([w[i, t] for i in data.Ig[g]]) <= data.nMax[g, t]*y[t-1]
                    ,name='non_negligence_2[%s,%s]'%(g,t))
                    
    def __build_logical_1__(self):
        m = self.model
        data = self.data
        e  = self.variables.e
        s  = self.variables.s
        for i in data.I:
            m.addCons(
                sum([t*e[i, t] for t in data.T]) >= sum([t*s[i, t] for t in data.T])
                ,name='logical_1[%s]'%i)
    
    def __build_logical_2__(self):
        m = self.model
        data = self.data
        e  = self.variables.e
        for i in data.I:
            m.addCons(
                sum([e[i, t] for t in data.T]) <= 1
                ,name='logical_2[%s]'%i)
    
    def __build_logical_3__(self):
        m = self.model
        data = self.data
        r  = self.variables.r
        tr  = self.variables.tr
        u  = self.variables.u
        for i in data.I:
            for t in data.T:
                m.addCons(
                    r[i, t] + tr[i, t] <= u[i, t]
                    ,name='logical_3[%s,%s]'%(i,t))
    
    def __build_logical_4__(self):
        m = self.model
        data = self.data
        w  = self.variables.w
        z  = self.variables.z
        for i in data.I:
            m.addCons(
                sum([w[i, t] for t in data.T]) >= z[i]
                ,name='logical_4[%s]'%i)
            
    def __build_valid_y__(self):
        m = self.model
        data = self.data
        y  = self.variables.y
        for t in data.T:
            m.addCons(y[t-1] >= y[t], name='valid_y[%s]'%t)
            
        m.addCons(y[7] >= 1, name='prueba_y')
        
        s  = self.variables.s
        m.addCons(s['aircraft_1',1] >= 1, name='prueba_start_1')
            
    def __build_valid_work_y__(self):
        m = self.model
        data = self.data
        w  = self.variables.w
        y  = self.variables.y
        for i in data.I:
            for t in data.T:
                m.addCons(
                    w[i, t] <= y[t-1]
                    ,name='valid_work_y[%s,%s]'%(i,t))
            
    def __build_constraints_all__(self):
        #self.__build_wildfire_containment_0__()
        self.__build_wildfire_containment_1__()
        self.__build_wildfire_containment_2__()
        self.__build_start_activity_1__()
        self.__build_start_activity_2__()
        self.__build_start_activity_3__()
        self.__build_end_activity__()
        self.__build_breaks_1__()
        self.__build_breaks_2__()
        self.__build_breaks_3__()
        self.__build_breaks_4__()
        self.__build_max_usage_periods__()
        self.__build_non_negligence_1__()
        self.__build_non_negligence_2__()
        self.__build_logical_1__()
        self.__build_logical_2__()
        self.__build_logical_3__()
        self.__build_logical_4__()
        self.__build_valid_y__()
        self.__build_valid_work_y__()
        
    def __build_constraints_master__(self):
        self.__build_start_activity_2__()
        self.__build_start_activity_3__()
        self.__build_max_usage_periods__()
        self.__build_logical_1__()
        self.__build_logical_2__()
        
    def __build_constraints_subproblem__(self):
        #self.__build_wildfire_containment_0__()
        self.__build_wildfire_containment_1__()
        self.__build_wildfire_containment_2__()
        self.__build_start_activity_1__()
        self.__build_end_activity__()
        self.__build_breaks_1__()
        self.__build_breaks_2__()
        self.__build_breaks_3__()
        self.__build_breaks_4__()
        self.__build_non_negligence_1__()
        self.__build_non_negligence_2__()
        self.__build_logical_3__()
        self.__build_logical_4__()
        
    def __build_constraints_subproblem_pricer__(self):
        #self.__build_wildfire_containment_0__()
        self.__build_start_activity_1__()
        self.__build_start_activity_2__()
        self.__build_start_activity_3__()
        self.__build_end_activity__()
        self.__build_breaks_1__()
        self.__build_breaks_2__()
        self.__build_breaks_3__()
        self.__build_breaks_4__()
        self.__build_max_usage_periods__()
        self.__build_logical_1__()
        self.__build_logical_2__()
        self.__build_logical_3__()
        self.__build_logical_4__()
        self.__build_valid_y__()
        self.__build_valid_work_y__()
        
    def get_original_model(self):
        self.model = scip.Model("Master")
        self.__build_variables_all__()
        self.__build_objective_all__()
        self.__build_constraints_all__()
        
        return self.model
        
    def get_master_model(self):
        self.model = scip.Model("Master")
        self.__build_variables_master__()
        self.__build_objective_master__()
        self.__build_constraints_master__()
        
        return self.model
        
    def get_subproblem_model(self):
        self.model = scip.Model("Subproblem")
        self.__build_variables_subproblem__()
        self.__build_objective_subproblem__()
        self.__build_constraints_subproblem__()
        
        return self.model
    
    def get_subproblem_pricer_model(self):
        self.model = scip.Model("Subproblempricer")
        self.__build_variables_all__()
        self.__build_objective_subproblem_pricer__()
        self.__build_constraints_subproblem_pricer__()
        
        return self.model
    
    def get_solution_variables_all(self):
        
        m = self.model
        data = self.data
        
        s = {(i,t): m.getVal(self.variables.s[i,t]) for i in data.I for t in data.T}
        tr = {(i,t): m.getVal(self.variables.tr[i,t]) for i in data.I for t in data.T}
        r = {(i,t): m.getVal(self.variables.r[i,t]) for i in data.I for t in data.T}
        er = {(i,t): m.getVal(self.variables.er[i,t]) for i in data.I for t in data.T}
        e = {(i,t): m.getVal(self.variables.e[i,t]) for i in data.I for t in data.T}
        u = {(i, t): sum([s[i,tind] for tind in data.T_int(p_max=t)])
                   - sum([e[i,tind] for tind in data.T_int(p_max=t-1)])
            for i in data.I for t in data.T}
        w = {(i, t): u[i, t] - r[i, t] - tr[i, t] for i in data.I for t in data.T}
        z = {i: sum([e[i,tind] for tind in data.T]) for i in data.I}
        y = {t: m.getVal(self.variables.y[t]) for t in data.T}
        mu = {(g,t): m.getVal(self.variables.mu[g,t]) for g in data.G for t in data.T}
        
        return {'s':s,
                'tr':tr,
                'r':r,
                'er':er,
                'e':e,
                'u':u,
                'w':w,
                'z':z,
                'y':y,
                'mu':mu}
        
    def write_decompositionfile(self,file_name):
        
        out_file = open(file_name,'w')
        data = self.data
        
        out_file.write('Presolved \n')
        out_file.write(str(0)+'\n')
        
        out_file.write('NBlocks \n')
        out_file.write(str(len(data.I))+'\n')
        
        ###################################
        # Constraints divided by resource
        for counter,i in enumerate(data.I):
            bloque = 'Block %s'%str(counter+1)
            out_file.write(bloque + '\n')
            cons_name = []
            for t in data.T:
                cons_name.append('start_activity_1[%s,%s]'%(i,t))
                cons_name.append('end_activity[%s,%s]'%(i,t))
                cons_name.append('Breaks_1_lb[%s,%s]'%(i,t))
                cons_name.append('Breaks_1_ub[%s,%s]'%(i,t))
                cons_name.append('Breaks_2[%s,%s]'%(i,t))
                cons_name.append('Breaks_3[%s,%s]'%(i,t))
                cons_name.append('Breaks_4[%s,%s]'%(i,t))
                cons_name.append('logical_3[%s,%s]'%(i,t))
                
            if data.ITW[i] == True:
                cons_name.append('start_activity_2[%s]'%i)
            else:
                cons_name.append('start_activity_3[%s]'%i)
                
            cons_name.append('max_usage_periods[%s]'%i)
            cons_name.append('logical_1[%s]'%i)
            cons_name.append('logical_2[%s]'%i)
            cons_name.append('logical_4[%s]'%i)
                
            for con in cons_name:
                out_file.write(con + '\n')
        
        ######################
        # Linking constraints
        out_file.write('Masterconss\n')
        # Wildfire Contraints
        out_file.write('wildfire_containment_1'+'\n')
        for t in data.T:
            con_name='wildfire_containment_2[%s]'%(t)
            out_file.write(con_name+'\n')
        
        # Non negligence constraints
        for g in data.G:
            for t in data.T:
                con_name1 ='non_negligence_1[%s,%s]'%(g,t)
                con_name2 ='non_negligence_2[%s,%s]'%(g,t)
                out_file.write(con_name1+'\n')
                out_file.write(con_name2+'\n')
 
        out_file.close()
        
        
    def writeGCGfiles(self, model_file = 'firedecomp.mps', decomp_file = 'firedecomp.dec'):
        
        # Formulating the model:
        model = self.get_original_model()
        
        # Write mps file
        model.writeProblem(model_file)
        
        # Write decomposition file:
        self.write_decompositionfile(decomp_file)
        

def solve_original(problem_data,solver_options):
    
    # Creating the model
    original_problem = Problem_model(problem_data.data)
    original = original_problem.get_original_model()
    
    # Defining the options:
    if solver_options is None:
        solver_options = {'display/verblevel': 0}

    # set scip options: https://scip.zib.de/doc/html/PARAMETERS.php
    if isinstance(solver_options, dict):
        for k, v in solver_options.items():
            master.setParam(k, v)

    # optimizing the problem using original formulation
    original.optimize()
    
    #original.writeProblem('originalModel.lp')
    #original.writeBestSol('original.sol')

    # Todo: check what status number return a solution
    status = original.getStatus()
    if status != "infeasible":
        #print('Objective Function = ',original.getObjVal())
        
        data = problem_data.data
        # Load variables values
        problem_data.resources.update(
            {i: {'select': sum([original.getVal(original_problem.variables.e[i,t]) for t in data.T])== 1}
             for i in data.I})
             
        uval = {(i,t): sum([original.getVal(original_problem.variables.s[i, tind]) for tind in data.T_int(p_max=t)])
                  - sum([original.getVal(original_problem.variables.e[i, tind]) for tind in data.T_int(p_max=t-1)])
                for i in data.I for t in data.T}
        wval = {(i,t): uval[i,t] - original.getVal(original_problem.variables.r[i, t]) - original.getVal(original_problem.variables.tr[i, t])
                for i in data.I for t in data.T}
        
        problem_data.resources_wildfire.update(
            {(i, t): {
                'start': original.getVal(original_problem.variables.s[i, t]) == 1,
                'travel': original.getVal(original_problem.variables.tr[i, t]) == 1,
                'rest': original.getVal(original_problem.variables.r[i, t]) == 1,
                'end_rest': original.getVal(original_problem.variables.er[i, t]) == 1,
                'end': original.getVal(original_problem.variables.e[i, t]) == 1,
                'use':  uval[i,t] == 1,
                'work': wval[i, t] == 1
            }
             for i in data.I for t in data.T})

        problem_data.groups_wildfire.update(
            {(g, t): {'num_left_resources': original.getVal(original_problem.variables.mu[g, t])}
             for g in data.G for t in data.T})

        contained = {t: original.getVal(original_problem.variables.y[t]) == 0 for t in data.T}
        contained_period = [t for t, v in contained.items()
                            if v is True]

        if len(contained_period) > 0:
            first_contained = min(contained_period) + 1
        else:
            first_contained = data.max_t + 1

        problem_data.wildfire.update(
            {t: {'contained': False if t < first_contained else True}
             for t in data.T})
    else:
        print("SCIP status is: ",status)
        
    
    return original_problem, status     
    
def solve_benders(problem_data, solver_options):
    
    # original_problem = Problem_model(problem_data.data)
    # original = original_problem.get_original_model()
    # original.optimize()
    # print('Original Objective Function = ',original.getObjVal())
    
    master_problem = Problem_model(problem_data.data)
    master = master_problem.get_master_model()
    
    subprob_problem = Problem_model(problem_data.data)
    subprob = subprob_problem.get_subproblem_model()
    
    # initializing the default Benders' decomposition with the subproblem
    master.setPresolve(scip.SCIP_PARAMSETTING.OFF)
    master.setBoolParam("misc/allowdualreds", False)
    master.setBoolParam("benders/copybenders", False)
    master.initBendersDefault(subprob)
    
    # Defining the options:
    if solver_options is None:
        solver_options = {'display/verblevel': 0}

    # set scip options: https://scip.zib.de/doc/html/PARAMETERS.php
    if isinstance(solver_options, dict):
        for k, v in solver_options.items():
            master.setParam(k, v)

    # optimizing the problem using Benders' decomposition
    master.optimize()

    # solving the subproblems to get the best solution
    master.computeBestSolSubproblems()
    
    master.printStatistics()
    
    # Todo: check what status number return a solution
    status = master.getStatus()
    if status != "infeasible":
        #print('Master Objective Function = ',master.getObjVal())
        #print('Subproblem Objective Function = ',subprob.getObjVal())
        
        data = problem_data.data
        # Load variables values
        problem_data.resources.update(
            {i: {'select': sum([master.getVal(master_problem.variables.e[i,t]) for t in data.T])== 1}
             for i in data.I})
             
        uval = {(i,t): sum([master.getVal(master_problem.variables.s[i, tind]) for tind in data.T_int(p_max=t)])
                  - sum([master.getVal(master_problem.variables.e[i, tind]) for tind in data.T_int(p_max=t-1)])
                for i in data.I for t in data.T}
        zval = {(i,t): uval[i,t] - subprob.getVal(subprob_problem.variables.r[i, t]) - subprob.getVal(subprob_problem.variables.tr[i, t])
                for i in data.I for t in data.T}
        
        problem_data.resources_wildfire.update(
            {(i, t): {
                'start': master.getVal(master_problem.variables.s[i, t]) == 1,
                'travel': subprob.getVal(subprob_problem.variables.tr[i, t]) == 1,
                'rest': subprob.getVal(subprob_problem.variables.r[i, t]) == 1,
                'end_rest': subprob.getVal(subprob_problem.variables.er[i, t]) == 1,
                'end': master.getVal(master_problem.variables.e[i, t]) == 1,
                'use':  uval[i,t] == 1,
                'work': zval[i, t] == 1
            }
             for i in data.I for t in data.T})

        problem_data.groups_wildfire.update(
            {(g, t): {'num_left_resources': subprob.getVal(subprob_problem.variables.mu[g, t])}
             for g in data.G for t in data.T})

        contained = {t: subprob.getVal(subprob_problem.variables.y[t]) == 0 for t in data.T}
        contained_period = [t for t, v in contained.items()
                            if v is True]

        if len(contained_period) > 0:
            first_contained = min(contained_period) + 1
        else:
            first_contained = data.max_t + 1

        problem_data.wildfire.update(
            {t: {'contained': False if t < first_contained else True}
             for t in data.T})
    else:
        print("SCIP status is: ",status)
        
    return master_problem, status

        
def solve_GCG(problem_data, model_name = 'fireproblem', solver_options=None):

    # Create temporal scip folder
    auxiliary_folder = os.getcwd()+"/temp_scip"
    if not os.path.exists(auxiliary_folder):
        os.makedirs(auxiliary_folder)

    model_file = auxiliary_folder + "/" + model_name + '.mps'
    decomp_file = auxiliary_folder + "/" + model_name + '_decomp.dec'
    sol_file = auxiliary_folder + "/" + 'sol_' + model_name + '.sol'
    options_file = auxiliary_folder + "/" + model_name + 'scip_options.set'

    # Creating the model:
    original_problem = Problem_model(problem_data.data)

    # Write gcc input files (model and decomposition)
    original_problem.writeGCGfiles(model_file, decomp_file)

    # Write options to a file:
    if solver_options is None:
        solver_options = {'display/verblevel': 0}

    with open(options_file, "w") as scip_options:
      for option in solver_options:
        scip_options.write(str(option) + '= ' + str(solver_options[option]) + "\n")

    # Call to system:
    gcg_commands  = 'set load ' + options_file
    gcg_commands += ' r ' + model_file
    gcg_commands += ' r ' + decomp_file
    gcg_commands += ' optimize '
    gcg_commands += ' write solution ' + sol_file
    gcg_commands += ' q'
        
    call_line = "./" + GCG_DIR + "/gcg" + " -c " + "\'" + gcg_commands + "\'"
    status = subprocess.call(call_line, shell=True)
    
    # Check if the solution file was generated:
    exists = os.path.isfile(sol_file)
    if exists:
        readSolFileAndUpdate(sol_file,problem_data)
    else:
        status = -1
        
    return status
    
def readSolFileAndUpdate(sol_file, problem):
        
    data = problem.data
    
    regexp = re.compile(r"^(\S*)\[(\S*)\]\s*(\S*)",re.VERBOSE)
    var_name = []
    var_key = []
    var_val = []
    with open(sol_file,'r') as sf:
        for counter, line in enumerate(sf):
            if counter>1:
                sol_info = re.search(regexp,line)
                var_name.append(sol_info.group(1))
                var_key.append(sol_info.group(2))
                var_val.append(sol_info.group(3))
    
    resources_wildfire = {(i,t):
        {
        'start': False,
        'travel': False,
        'rest': False,
        'end_rest': False,
        'end': False,
        'use':  False,
        'work': False
        }
        for i in data.I for t in data.T}
        
    contained = {t:False for t in data.T}
    groups_wildfire = {(g,t): 
        {
        'num_left_resources': False
        }
        for g in data.G for t in data.T}
    
    for counter,var in enumerate(var_name):
        key = var_key[counter]
        val = float(var_val[counter])
        if var == 'contention':
            t = int(key)
            if val == 0:
                contained[t] = True
        elif var == 'missing_resources':
            g,t = key.split(',')
            t = int(t)
            if val == 0:
                groups_wildfire[g,t]['num_left_resources'] = True
        else: # 'start, travel, rest, '
            i,t = key.split(',')
            t = int(t)
            if val == 1:
                resources_wildfire[(i,t)][var] = True
    
    # Define z, u and w variable:
    resources = {i: 
                {
                'select': sum([resources_wildfire[i,t]['end'] for t in data.T])>0
                }
                for i in data.I}
    
    for i in data.I:
        for t in data.T:
            uval = (sum([resources_wildfire[(i,tind)]['start'] for tind in data.T_int(p_max=t)])
                  - sum([resources_wildfire[(i,tind)]['end'] for tind in data.T_int(p_max=t-1)]))
            wval = uval - resources_wildfire[(i,t)]['rest'] - resources_wildfire[(i,t)]['travel']
            if uval > 0:
                resources_wildfire[(i,t)]['use'] = True
            if wval > 0:
                resources_wildfire[(i,t)]['work'] = True
            
    # Update problem objects:
    problem.resources.update(resources)
    problem.resources_wildfire.update(resources_wildfire)
    problem.groups_wildfire.update(groups_wildfire)
    
    contained_period = [t for t, v in contained.items() if v is True]
    if len(contained_period) > 0:
        first_contained = min(contained_period) + 1
    else:
        first_contained = data.max_t + 1
        
    problem.wildfire.update(
        {t: {'contained': False if t < first_contained else True}
         for t in data.T})
    
# --------------------------------------------------------------------------- #

