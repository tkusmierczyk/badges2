#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import minimize
import sys
sys.path.append("../")

from clustering.poisson_ll import PoissonLL


from matplotlib import pyplot
from scipy.stats import gamma


def _f1(f):
    if f is None: return "None"
    if type(f)==float: return "%.1f" % f
    return f

def _plot_gamma_(rng, shape, scale, **kwargs):
    #if logging.getLogger("output").getEffectiveLevel()==logging.DEBUG:
    rng = int(np.ceil(rng))
    rv = gamma(shape, scale=scale)
    xs = (np.array(range(rng*20)))/10
    ys = rv.pdf(xs)
    pyplot.plot(xs, ys, **kwargs)
        
                
def _report_poisson_fitting(fit, label="fit"):
    fit_str = " ".join(e for e in str(fit).replace("\n", ", ").replace("\t", " ").split(" ") if e!="")
    logging.debug("%s %s" % (label, fit_str))
    shape0, rate0 = fit.x
    scale0 = 1.0 / rate0
    mean0, stddev0 = shape0 * scale0, np.sqrt(shape0) * scale0
    logging.debug("%s mean=%.3f std=%.3f shp=%.3f scale=%.3f" % (label, mean0, stddev0, shape0, scale0))
    _plot_gamma_(mean0+3*stddev0, shape0, scale0, lw=2,
                 label="%s\nm=%.3f s=%.3f shp=%.3f scl=%.3f" % (label, mean0, stddev0, shape0, scale0))


def _report_twopoisson_fitting(fit, fit0, fit1, label=""):
    #show overall fitting results
    if logging.getLogger("output").getEffectiveLevel()>logging.DEBUG: return
    pyplot.cla()
    _report_poisson_fitting(fit,  label+"[fit ]")
    _report_poisson_fitting(fit0, label+"[fit0]")
    _report_poisson_fitting(fit1, label+"[fit1]")
    pyplot.grid(True)
    pyplot.legend(fontsize=8)
    pyplot.savefig("/tmp/poisson_ll_fitted.png")


def s(x):
    return 1 / (1 + np.exp(-x))


class TwoPoissonEM:
    """
    Clustering of Poisson processes with 
    priors (pi) being a function (logit) of covariates. 
    """
        
    def __init__(self, users0, users1, features, K=2):
        """
            Args:
                users0 - a list of tuples (count, lifetime) representing users in the first part  
                users1 - a list of tuples (count, lifetime) representing users in the second part 
        """
        assert len(users0)==len(users1)
        assert len(users0)==len(features)
        self.users0 = users0
        self.users1 = users1
        self.users = [(n0+n1, lt0+lt1) for (n0, lt0), (n1, lt1) in zip(users0, users1)]
        self.features = features #@TODO normalization?

        self.weights = None 

        z = np.random.uniform(size=(len(self.users), K))
        norm = z.sum(1)        
        self.z = np.divide(z, np.array([norm for _ in range(K)]).T)
                        
        self.a = None
        self.b = None
        self.a0 = None
        self.b0 = None
        self.a1 = None
        self.b1 = None
        self.pi = None
        
        #store for debugging
        self.pf = None 
        self.m_count = 0
        self.e_count = 0
        
                
    def predict(self, features):
        N = features.shape[0]
        assert features.shape[1]==len(self.weights)
        return [s(np.inner(features[u, :], self.weights)) for u in range(N)]
        
    def maximize_pi(self):
        #Nk = self.z.sum(0)
        #self.pi = Nk / N #@TODO should we include pi in numerical optimization ???
        
        N = len(self.users)        
        def err(weights):
            return sum((self.z[u,1]-s(np.inner(self.features[u, :], weights)))**2 for u in range(N)) / float(N)
        
        if self.weights is None:
            D = self.features.shape[1]        
            self.weights = [np.random.randn() for _ in range(D)] #@TODO better init
                 
        wfit = minimize(err, 
                       self.weights, 
                       #jac=lambda pos: -pf.drv_LL(pos), 
                       method="BFGS", 
                       options={'gtol': 1e-15, 'disp': False})
        self.weights = wfit.x
        logging.debug("[twopoisson_em][#E=%s,#M=%s][M][maximize_pi] weights=%s" % 
                      (self.e_count, self.m_count, self.weights))
        
        self.pi = np.zeros((N,2))
        self.pi[:,1] = [s(np.inner(self.features[u, :], self.weights)) for u in range(N)]
        self.pi[:,0] = 1.0-self.pi[:,1]
        
    def gamma_fitting(self, users, 
                      pi=None, z=None, init_pos=None,
                      fitting_id=0):
        self.pf = pf = PoissonLL(users, pi, z)    
        if init_pos is None:
            init_pos = np.array([np.random.uniform()*100, np.random.uniform()*100])
        cons = [ {'type': 'ineq', 'fun': pf.con_a}, {'type': 'ineq', 'fun': pf.con_b} ]
        fit = minimize(lambda pos: -pf.LL(pos), 
                       init_pos, 
                       jac=lambda pos: -pf.drv_LL(pos), 
                       method="COBYLA", 
                       constraints=cons,
                       options={'disp': False}) #options={'gtol': 1e-15, 'disp': False}, callback=pf.callback)
        
        logging.debug("[twopoisson_em][#E=%s,#M=%s][M][gamma_fitting-run%s] fitting %i users, init_a&b=%s sum(z)=%s -> LLopt=%s a&b_opt=%s" % 
                      (self.e_count, self.m_count, fitting_id,
                       len(users), init_pos, ("None" if z is None else sum(z)), _f1(-fit.fun), fit.x))
        return fit
    
    def gamma_fitting_multiple_start(self, users, 
                                     pi=None, z=None, init_pos=None, numstarts=3):
        if init_pos is not None: numstarts=1     
        fits = [self.gamma_fitting(users, pi, z, init_pos, i) for i in range(numstarts)]
        fits = sorted(fits, key=lambda fit: fit.fun)
        fit  = fits[0]    
        logging.debug("[twopoisson_em][#E=%s,#M=%s][M][gamma_fitting-sel ] selected: LLopt=%s a&b_opt=%s" % 
                      (self.e_count, self.m_count,  _f1(-fit.fun), fit.x))        
        return fit         
                
    def maximization(self):
        self.maximize_pi()
        
        #fitting = self.gamma_fitting
        fitting = self.gamma_fitting_multiple_start 

        logging.debug("[twopoisson_em][#E=%s,#M=%s][M][fit - fitting gamma for whole data]:" % (self.e_count, self.m_count))        
        fit  = fitting(self.users,  pi=self.pi[:,0], z=self.z[:,0], init_pos=((self.a, self.b) if self.a else None))
        self.a,  self.b  = fit.x
        
        logging.debug("[twopoisson_em][#E=%s,#M=%s][M][fit0 - fitting gamma for users before badge]:" % (self.e_count, self.m_count))
        fit0 = fitting(self.users0, pi=self.pi[:,1], z=self.z[:,1], init_pos=((self.a0, self.b0) if self.a0 else None))
        self.a0, self.b0 = fit0.x
        
        logging.debug("[twopoisson_em][#E=%s,#M=%s][M][fit1 - fitting gamma for users after badge]:" % (self.e_count, self.m_count))
        fit1 = fitting(self.users1, pi=self.pi[:,1], z=self.z[:,1], init_pos=((self.a1, self.b1) if self.a1 else None))
        self.a1, self.b1 = fit1.x
        
        _report_twopoisson_fitting(fit, fit0, fit1, label=("[twopoisson_em][#E=%s,#M=%s][M][result]" % (self.e_count, self.m_count)) )
        self.m_count += 1
                                                    
    def expectation(self):  
        #update assignments 
        logging.debug("[twopoisson_em][#E=%s,#M=%s][E]:" % (self.e_count, self.m_count))     
        N = len(self.users)
        for u in range(N): 
            n, lt   = self.users[u]
            n0, lt0 = self.users0[u]
            n1, lt1 = self.users1[u]

            LL   = PoissonLL.user_LL(self.a,  self.b,  n,  lt) + np.log(self.pi[u, 0])
            LL01 = PoissonLL.user_LL(self.a0, self.b0, n0, lt0) + PoissonLL.user_LL(self.a1, self.b1, n1, lt1) + np.log(self.pi[u, 1])
                    
            self.z[u, 0] = np.exp(LL - logsumexp([LL, LL01]))
            self.z[u, 1] = np.exp(LL01 - logsumexp([LL, LL01]))
            
            #logging.debug("[uid=%4s] n=%5s lt=%6s n0=%5s lt0=%6s n1=%5s lt1=%6s  LL=%7s LL01=%7s LL0=%7s LL1=%7s => %s" %
            #          (u, n, "%.1f" % lt, n0, "%.1f" % lt0, n1, "%.1f" % lt1, 
            #           "%.1f" % LL, "%.1f" % (LL0+LL1), "%.1f" % LL0, "%.1f" % LL1, ("H0" if LL>LL0+LL1 else "H1")))
            
        #renomalize to remove numeric effects
        K = self.z.shape[1]
        norm = self.z.sum(1)
        self.z = np.divide(self.z, np.array([norm for _ in range(K)]).T)
        self.e_count += 1
        
    def LL(self):
        N = len(self.users)
        total = 0
        for u in range(N): 
            n, lt   = self.users[u]
            n0, lt0 = self.users0[u]
            n1, lt1 = self.users1[u]

            LL   = PoissonLL.user_LL(self.a,  self.b,  n,  lt) + np.log(self.pi[u, 0]) 
            LL01 = PoissonLL.user_LL(self.a0, self.b0, n0, lt0) + PoissonLL.user_LL(self.a1, self.b1, n1, lt1) + np.log(self.pi[u, 1])   
            
            total += logsumexp([LL, LL01])
            
        return float(total)
        
    def positives_proportion(self):
        return sum(self.positives())/len(self.positives())

    def positives(self):
        return self.z[:,1]

