#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
from scipy.special import loggamma, digamma
#from scipy.misc import logsumexp



class PoissonLL: 
    """Log-Likelihood calculation of collapsed Poisson Processes with Gamma priors."""

    def __init__(self, users, pi=None, z=None):
        self.users = users
        
        if pi is None: pi = 1.0
        self.pi = pi if hasattr(pi, '__len__') else [pi for _ in users]
        
        if z is None: z = 1.0
        self.z = z if hasattr(z, '__len__') else [z for _ in users]
        
        assert len(self.z)==len(self.users)
        assert len(self.pi)==len(self.users)         

    @staticmethod
    def user_LL(a, b, n, lt):
        log = np.log
        return (a*log(b) - (a+n)*log(lt+b) + loggamma(a+n) - loggamma(a) )
    
    def LL(self, params):
        a, b = params
        vs = [zu * (PoissonLL.user_LL(a, b, n, lt) + np.log(piu)) for (n, lt), zu, piu in zip(self.users, self.z, self.pi)]
        return sum(vs)  
                
    def drv_LL(self, params):
        log = np.log
        a, b = params
        
        vs = [zu * (log(b) - log(lt+b) + digamma(a+n) - digamma(a) ) for (n, lt), zu in zip(self.users, self.z)]
        drvf_a = sum(vs)

        vs = [zu * ((a/b) - (a+n)/(lt+b)) for (n, lt), zu in zip(self.users, self.z)]
        drvf_b = sum(vs)
        
        return np.array([drvf_a, drvf_b])       

    def con_a(self, params):
        a, _ = params
        return a-0.000001
        
    def con_b(self, params):
        _, b = params
        return b-0.000001
        
    def callback(self, xk):
        logging.debug("[PoissonLL] optimization callback: current parameters value=%s" % xk)
        


################################################################
################################################################
################################################################

        


def report_twopoisson_fitting_users(users0, users1, fit, fit0, fit1): #TODO     
    """Per user fitting results."""
    
    users = [(n0+n1, lt0+lt1) for (n0, lt0), (n1, lt1) in zip(users0, users1)]
    logging.debug("%i users" % len(users))
    for u in range(len(users)):
        n, lt   = users[u]
        n0, lt0 = users0[u]
        n1, lt1 = users1[u]
        
        LL = PoissonLL.user_LL(fit.x[0], fit.x[1], n, lt)
        LL0 = PoissonLL.user_LL(fit0.x[0], fit0.x[1], n0, lt0)
        LL1 = PoissonLL.user_LL(fit1.x[0], fit1.x[1], n1, lt1)
        logging.debug("[poisson_ll][uid=%4s] n=%5s lt=%6s n0=%5s lt0=%6s n1=%5s lt1=%6s  LL=%7s LL01=%7s LL0=%7s LL1=%7s => %s" %
                      (u, n, "%.1f" % lt, n0, "%.1f" % lt0, n1, "%.1f" % lt1, 
                       "%.1f" % LL, "%.1f" % (LL0+LL1), "%.1f" % LL0, "%.1f" % LL1, ("H0" if LL>LL0+LL1 else "H1")))
    #report_twopoisson_fitting(fit, fit0, fit1)
    
